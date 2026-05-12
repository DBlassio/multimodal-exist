"""
Inference Script

Generates raw probability predictions for 3 models on the test set.
Each model uses fold-level averaging (fold ensemble). Then a cross-model
ensemble is computed.

Output: inference/predictions/{model}_raw.parquet with columns:
  id, lang, p21, p22_direct, p22_judgemental,
  p23_ideological, p23_stereotyping, p23_objectification,
  p23_sexual_violence, p23_misogyny

Usage:
  python run_inference.py \
      --test_parquet  data/processed/test_model_ready.parquet \
      --img_dir       data/memes/test/memes \
      --ckpt_dir      inference/checkpoints \
      --output_dir    inference/predictions
"""

import os, argparse, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel,AlignProcessor, AlignModel
warnings.filterwarnings("ignore")


TEXT_MODEL_NAME  = "microsoft/mdeberta-v3-base"
ALIGN_MODEL_NAME = "kakaobrain/align-base"
MAX_TEXT_LENGTH  = 256
COMMON_DIM       = 768
BATCH_SIZE       = 32         
GATE_TEMPERATURE = 0.3
N_ATTN_HEADS     = 8

TASK23_COLS = [
    "CAT_ideological_inequality",
    "CAT_misogyny_non_sexual_violence",
    "CAT_objectification",
    "CAT_sexual_violence",
    "CAT_stereotyping_dominance",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

#Model Architectures
class PhysioMLP(nn.Module):
    def __init__(self, input_dim, common_dim, dropout=0.1):
        super().__init__()
        hidden = common_dim // 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, common_dim), nn.LayerNorm(common_dim),
            nn.ReLU(), nn.Dropout(dropout))
    def forward(self, x):
        return self.net(x)


class EarlyFusionModel(nn.Module):
    def __init__(self, text_model_name, vision_model_name,
                 physio_dim, num_cat=5, common_dim=768, dropout=0.1):
        super().__init__()
        
        #Text
        self.text_encoder    = AutoModel.from_pretrained(text_model_name)
        text_dim             = self.text_encoder.config.hidden_size
        self.text_projection = (nn.Linear(text_dim, common_dim) if text_dim != common_dim else nn.Identity())
        
        #Vision
        align_full = AlignModel.from_pretrained(vision_model_name)
        self.vision_encoder = align_full.vision_model
        vision_dim = align_full.config.vision_config.hidden_dim
        self.vision_projection = (nn.Linear(vision_dim, common_dim) if vision_dim != common_dim else nn.Identity())
        self.physio_encoder = PhysioMLP(physio_dim, common_dim, dropout)

        #Projection
        proj_dim = common_dim // 2
        self.text_proj   = nn.Linear(common_dim, proj_dim)
        self.vision_proj = nn.Linear(common_dim, proj_dim)
        self.physio_proj = nn.Linear(common_dim, proj_dim)

        self.fusion_layer = nn.Sequential(
            nn.Linear(proj_dim * 3, common_dim), 
            nn.LayerNorm(common_dim),
            nn.ReLU(), nn.Dropout(dropout))

        self.head_21 = nn.Sequential(
            nn.Linear(common_dim, common_dim // 2), 
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(common_dim // 2, 2))
        
        self.head_22 = nn.Sequential(
            nn.Linear(common_dim, common_dim // 2), 
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(common_dim // 2, common_dim // 4),
            nn.ReLU(), nn.Dropout(dropout), 
            nn.Linear(common_dim // 4, 2))
        
        self.head_23 = nn.Sequential(
            nn.Linear(common_dim, common_dim // 2), 
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(common_dim // 2, common_dim // 4),
            nn.ReLU(), 
            nn.Dropout(dropout), nn.Linear(common_dim // 4, num_cat))

    def forward(self, input_ids, attention_mask, pixel_values, physio):
        #Text Embedding
        text_emb   = self.text_projection(self.text_encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state[:, 0, :])
        
        #Vision Embedding
        vision_emb = self.vision_projection(self.vision_encoder(pixel_values=pixel_values).pooler_output)
        
        #Physio Embedding
        physio_emb = self.physio_encoder(physio.float())

        #Fused
        fused = self.fusion_layer(torch.cat([
            self.text_proj(text_emb),
            self.vision_proj(vision_emb),
            self.physio_proj(physio_emb)], dim=1))
        
        return self.head_21(fused), self.head_22(fused), self.head_23(fused)


class GatedFusionModel(nn.Module):
    def __init__(self, text_model_name, vision_model_name,
                 eeg_dim, et_dim, num_cat=5, common_dim=768, dropout=0.1):
        super().__init__()

        #Text model
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.hidden_size
        self.text_projection = (nn.Linear(text_dim, common_dim) if text_dim != common_dim else nn.Identity())

        #Vision Model
        align_full = AlignModel.from_pretrained(vision_model_name)
        self.vision_encoder = align_full.vision_model
        vision_dim = align_full.config.vision_config.hidden_dim
        self.vision_projection = (nn.Linear(vision_dim, common_dim) if vision_dim != common_dim else nn.Identity())

        self.eeg_encoder = PhysioMLP(eeg_dim, common_dim, dropout)
        self.et_encoder  = PhysioMLP(et_dim,  common_dim, dropout)

        self.gate_vision = nn.Linear(common_dim, 1)
        self.gate_eeg    = nn.Linear(common_dim, 1)
        self.gate_et     = nn.Linear(common_dim, 1)

        self.fusion_layer = nn.Sequential(
            nn.Linear(common_dim, common_dim), 
            nn.LayerNorm(common_dim),
            nn.ReLU(), 
            nn.Dropout(dropout))

        self.head_21 = nn.Sequential(
            nn.Linear(common_dim, common_dim // 2),
              nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(common_dim // 2, 2))
        
        self.head_22 = nn.Sequential(
            nn.Linear(common_dim, common_dim // 2), 
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(common_dim // 2, common_dim // 4),
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(common_dim // 4, 2))
        
        self.head_23 = nn.Sequential(
            nn.Linear(common_dim, common_dim // 2), 
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(common_dim // 2, common_dim // 4),
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(common_dim // 4, num_cat))

    def forward(self, input_ids, attention_mask, pixel_values, eeg, et):

        #Text Embeddings
        text_emb   = self.text_projection(self.text_encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state[:, 0, :])

        #Vision Embedding
        vision_emb = self.vision_projection(self.vision_encoder(pixel_values=pixel_values).pooler_output)
        
        #EEG Embedding
        eeg_emb = self.eeg_encoder(eeg.float())
        
        #ET Embedding
        et_emb  = self.et_encoder(et.float())

        #Gates
        beta    = torch.sigmoid(self.gate_vision(text_emb) / GATE_TEMPERATURE)
        alpha   = torch.sigmoid(self.gate_eeg(text_emb)    / GATE_TEMPERATURE)
        lambda_ = torch.sigmoid(self.gate_et(text_emb)     / GATE_TEMPERATURE)

        #Gate Fusion 
        z     = text_emb + beta * vision_emb + alpha * eeg_emb + lambda_ * et_emb
        fused = self.fusion_layer(z)
        
        return self.head_21(fused), self.head_22(fused), self.head_23(fused)


class CrossAttentionGatedModel(nn.Module):
    def __init__(self, text_model_name, vision_model_name,
                 eeg_dim, et_dim, num_cat=5, common_dim=768,
                 n_attn_heads=8, dropout=0.1):
        super().__init__()

        #Text Model
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.hidden_size
        self.text_projection = (nn.Linear(text_dim, common_dim) if text_dim != common_dim else nn.Identity())

        #Vision Model
        align_full = AlignModel.from_pretrained(vision_model_name)
        self.vision_encoder = align_full.vision_model
        vision_dim = align_full.config.vision_config.hidden_dim
        self.vision_projection = (nn.Linear(vision_dim, common_dim) if vision_dim != common_dim else nn.Identity())

        self.eeg_encoder = PhysioMLP(eeg_dim, common_dim, dropout)
        self.et_encoder  = PhysioMLP(et_dim,  common_dim, dropout)

        attn = lambda: nn.MultiheadAttention(common_dim, n_attn_heads, dropout=dropout, batch_first=True)
        
        
        self.cross_attn_text_to_image  = attn()
        self.cross_attn_image_to_text  = attn()
        self.cross_attn_text_to_physio = attn()
        self.cross_attn_physio_to_text = attn()

        self.ln_text   = nn.LayerNorm(common_dim)
        self.ln_vision = nn.LayerNorm(common_dim)
        self.ln_physio = nn.LayerNorm(common_dim)

        self.gate_vision = nn.Linear(common_dim, 1)
        self.gate_eeg    = nn.Linear(common_dim, 1)
        self.gate_et     = nn.Linear(common_dim, 1)

        self.fusion_layer = nn.Sequential(
            nn.Linear(common_dim, common_dim), 
            nn.LayerNorm(common_dim),
            nn.ReLU(), 
            nn.Dropout(dropout))

        self.head_21 = nn.Sequential(
            nn.Linear(common_dim, common_dim // 2),
              nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(common_dim // 2, 2))
        
        self.head_22 = nn.Sequential(
            nn.Linear(common_dim, common_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(common_dim // 2, common_dim // 4),
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(common_dim // 4, 2))
        
        self.head_23 = nn.Sequential(
            nn.Linear(common_dim, common_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(common_dim // 2, common_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(common_dim // 4, num_cat))

    def forward(self, input_ids, attention_mask, pixel_values, eeg, et):

        #Text Embedding
        text_emb   = self.text_projection(self.text_encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state[:, 0, :])
        
        #Vision Embedding
        vision_emb = self.vision_projection(self.vision_encoder(pixel_values=pixel_values).pooler_output)

        #EEG embedding
        eeg_emb = self.eeg_encoder(eeg.float())
        
        #ET Embedding
        et_emb  = self.et_encoder(et.float())

        t_seq = text_emb.unsqueeze(1)
        v_seq = vision_emb.unsqueeze(1)
        p_seq = torch.stack([eeg_emb, et_emb], dim=1)

        t_from_v, _ = self.cross_attn_text_to_image(t_seq, v_seq, v_seq)
        v_from_t, _ = self.cross_attn_image_to_text(v_seq, t_seq, t_seq)
        t_from_p, _ = self.cross_attn_text_to_physio(t_seq, p_seq, p_seq)
        p_from_t, _ = self.cross_attn_physio_to_text(p_seq, t_seq, t_seq)

        text_enriched   = self.ln_text(text_emb + t_from_v.squeeze(1) + t_from_p.squeeze(1))
        vision_attended = self.ln_vision(vision_emb + v_from_t.squeeze(1))
        eeg_attended    = self.ln_physio(eeg_emb + p_from_t[:, 0, :])
        et_attended     = self.ln_physio(et_emb  + p_from_t[:, 1, :])

        beta    = torch.sigmoid(self.gate_vision(text_enriched) / GATE_TEMPERATURE)
        alpha   = torch.sigmoid(self.gate_eeg(text_enriched)    / GATE_TEMPERATURE)
        lambda_ = torch.sigmoid(self.gate_et(text_enriched)     / GATE_TEMPERATURE)

        #Fusion
        z     = text_enriched + beta * vision_attended + alpha * eeg_attended + lambda_ * et_attended
        fused = self.fusion_layer(z)
        return self.head_21(fused), self.head_22(fused), self.head_23(fused)


# DATASET
class TestDataset(Dataset):
    def __init__(self, df, img_dir, tokenizer, image_processor,
                 eeg_cols, et_cols, max_length=256):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.eeg_cols = eeg_cols
        self.et_cols = et_cols
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text_enc = self.tokenizer(str(row["text"]), max_length=self.max_length,
                                  padding="max_length", 
                                  truncation=True, return_tensors="pt")

        img_path = os.path.join(self.img_dir, str(row["image_file"]))
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            print("Image loading failed for:", img_path)
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        img_enc = self.image_processor(images=image, return_tensors="pt")

        eeg = torch.tensor(row[self.eeg_cols].values.astype(np.float32))
        et  = torch.tensor(row[self.et_cols].values.astype(np.float32))

        return {
            "id": row["id"],
            "input_ids": text_enc["input_ids"].squeeze(0),
            "attention_mask": text_enc["attention_mask"].squeeze(0),
            "pixel_values": img_enc["pixel_values"].squeeze(0),
            "eeg": eeg,
            "et": et,
            "physio": torch.cat([eeg, et]),
        }


def collate_fn(batch):
    ids = [b["id"] for b in batch]
    return {
        "ids": ids,
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "eeg": torch.stack([b["eeg"] for b in batch]),
        "et": torch.stack([b["et"] for b in batch]),
        "physio": torch.stack([b["physio"] for b in batch]),
    }


# INFERENCE HELPERS

@torch.no_grad()
def run_single_checkpoint(model, loader, model_type):
    """
    Run inference with one checkpoint.
    Returns dict: id → (p21, p22_direct, p23_array)
    """
    model.eval()
    results = {}

    for batch in tqdm(loader, desc="    batches", leave=False):
        with autocast(dtype=torch.bfloat16):
            if model_type == "baseline":
                l21, l22, l23 = model(
                    batch["input_ids"].to(DEVICE),
                    batch["attention_mask"].to(DEVICE),
                    batch["pixel_values"].to(DEVICE),
                    batch["physio"].to(DEVICE))
            else:  #  In csae of gated or cross_attention
                l21, l22, l23 = model(
                    batch["input_ids"].to(DEVICE),
                    batch["attention_mask"].to(DEVICE),
                    batch["pixel_values"].to(DEVICE),
                    batch["eeg"].to(DEVICE),
                    batch["et"].to(DEVICE))

        p21 = F.softmax(l21.float(), dim=1)[:, 1].cpu().numpy()
        p22 = F.softmax(l22.float(), dim=1).cpu().numpy()          # (B, 2)
        p23 = torch.sigmoid(l23.float()).cpu().numpy()              # (B, 5)

        for i, mid in enumerate(batch["ids"]):
            results[mid] = (float(p21[i]), p22[i], p23[i])

    return results


def average_fold_results(fold_results_list):
    """Average predictions across folds for one model."""
    all_ids = list(fold_results_list[0].keys())
    averaged = {}
    for mid in all_ids:
        p21s = [r[mid][0] for r in fold_results_list]
        p22s = [r[mid][1] for r in fold_results_list]
        p23s = [r[mid][2] for r in fold_results_list]
        averaged[mid] = (
            float(np.mean(p21s)),
            np.mean(p22s, axis=0),
            np.mean(p23s, axis=0),
        )
    return averaged


def results_to_df(results_dict, meme_ids):
    """Convert results dict to DataFrame."""
    rows = []
    for mid in meme_ids:
        p21, p22, p23 = results_dict[mid]
        rows.append({
            "id": mid,
            "p21": p21,
            "p22_direct": float(p22[1]),       # DIRECT = class 1
            "p22_judgemental": float(p22[0]),       # JUDGEMENTAL = class 0
            "p23_ideological": float(p23[0]),
            "p23_misogyny": float(p23[1]),
            "p23_objectification": float(p23[2]),
            "p23_sexual_violence": float(p23[3]),
            "p23_stereotyping": float(p23[4]),
        })
    return pd.DataFrame(rows)


def load_model(model_type, eeg_dim, et_dim, physio_dim):
    """Instantiate the correct model architecture."""
    if model_type == "baseline":
        return EarlyFusionModel(TEXT_MODEL_NAME, ALIGN_MODEL_NAME, physio_dim=physio_dim, common_dim=COMMON_DIM)
    elif model_type == "gated":
        return GatedFusionModel(TEXT_MODEL_NAME, ALIGN_MODEL_NAME,eeg_dim=eeg_dim, et_dim=et_dim, common_dim=COMMON_DIM)
    elif model_type == "cross_attention":
        return CrossAttentionGatedModel(TEXT_MODEL_NAME, ALIGN_MODEL_NAME,eeg_dim=eeg_dim, et_dim=et_dim,n_attn_heads=N_ATTN_HEADS, common_dim=COMMON_DIM)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_parquet", default="data/processed/test_model_ready.parquet")
    parser.add_argument("--img_dir",      default="data/memes/test/memes")
    parser.add_argument("--ckpt_dir",     default="inference/checkpoints")
    parser.add_argument("--output_dir",   default="inference/predictions")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"\nLoading test parquet: {args.test_parquet}")
    df = pd.read_parquet(args.test_parquet)
    print(f"  Test memes: {len(df)}")

    EEG_COLS = sorted([c for c in df.columns if c.startswith("EEG_") and c not in {"EEG_n_users", "EEG_raw", "et_n_users"}])
    ET_COLS = sorted([c for c in df.columns if c.startswith("et_") and c not in {"EEG_n_users", "EEG_raw", "et_n_users"}])
    EEG_DIM = len(EEG_COLS)
    ET_DIM = len(ET_COLS)
    PHYSIO_DIM = EEG_DIM + ET_DIM
    print(f"  EEG_DIM={EEG_DIM}  ET_DIM={ET_DIM}  PHYSIO_DIM={PHYSIO_DIM}")

    # Tokenizer & Processor
    print(f"\nLoading tokenizer and ALIGN processor...")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    align_processor = AlignProcessor.from_pretrained(ALIGN_MODEL_NAME)

    #Dataset and Loadr
    dataset = TestDataset(df, args.img_dir, tokenizer, align_processor, EEG_COLS, ET_COLS, MAX_TEXT_LENGTH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)
    meme_ids = df["id"].tolist()

    #Model (Checkpoint to Folds)
    model_configs = {
        "baseline": ("baseline", sorted([f for f in os.listdir(os.path.join(args.ckpt_dir, "baseline")) if f.endswith(".pt")])),
        "gated": ("gated", sorted([f for f in os.listdir(os.path.join(args.ckpt_dir, "gated")) if f.endswith(".pt")])),
        "cross_attention":("cross_attention",sorted([f for f in os.listdir(os.path.join(args.ckpt_dir, "cross_attention")) if f.endswith(".pt")])),
    }

    all_model_results = {}

    for model_name, (model_type, ckpt_files) in model_configs.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}  ({len(ckpt_files)} checkpoints)")
        print(f"{'='*60}")

        fold_results = []
        for ckpt_file in ckpt_files:
            ckpt_path = os.path.join(args.ckpt_dir, model_name, ckpt_file)
            print(f"  Loading: {ckpt_file}")

            model = load_model(model_type, EEG_DIM, ET_DIM, PHYSIO_DIM).to(DEVICE).float()
            state = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(state)

            results = run_single_checkpoint(model, loader, model_type)
            fold_results.append(results)

            del model
            torch.cuda.empty_cache()

        # Average across folds
        avg_results = average_fold_results(fold_results)
        all_model_results[model_name] = avg_results

        # Save per-model predictions
        df_preds = results_to_df(avg_results, meme_ids)
        out_path = os.path.join(args.output_dir, f"{model_name}_raw.parquet")
        df_preds.to_parquet(out_path, index=False)
        print(f"  Saved to {out_path}")
        print(f"  p21 mean: {df_preds['p21'].mean():.4f}  "
              f"p21 > 0.5: {(df_preds['p21'] > 0.5).mean():.1%}")

    #Ensemble: Average across our 3 models
    print(f"\n{'='*60}")
    print("Computing ensemble (average of 3 models)")
    print(f"{'='*60}")

    ensemble = {}
    for mid in meme_ids:
        p21s = [all_model_results[m][mid][0] for m in all_model_results]
        p22s = [all_model_results[m][mid][1] for m in all_model_results]
        p23s = [all_model_results[m][mid][2] for m in all_model_results]
        ensemble[mid] = (
            float(np.mean(p21s)),
            np.mean(p22s, axis=0),
            np.mean(p23s, axis=0),
        )

    df_ensemble = results_to_df(ensemble, meme_ids)
    out_path = os.path.join(args.output_dir, "ensemble_raw.parquet")
    df_ensemble.to_parquet(out_path, index=False)
    print(f"  Saved to {out_path}")
    print(f"  p21 mean: {df_ensemble['p21'].mean():.4f}  "
          f"p21 > 0.5: {(df_ensemble['p21'] > 0.5).mean():.1%}")

    print("\n  Inference complete!")


if __name__ == "__main__":
    main()
