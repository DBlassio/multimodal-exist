"""
Data Loader — loads and serves all prediction data for the API.
All heavy data is loaded once at startup and kept in memory.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


T21 = 0.50

#Distribution based on our EDA (Train dataset)
TRAIN_FREQ = {
    "p23_ideological":     0.290,
    "p23_stereotyping":    0.225,
    "p23_objectification": 0.220,
    "p23_sexual_violence": 0.068,
    "p23_misogyny":        0.025,
}

# Mapping Category labels
CAT_COLS = {
    "IDEOLOGICAL-INEQUALITY":       "p23_ideological",
    "MISOGYNY-NON-SEXUAL-VIOLENCE": "p23_misogyny",
    "OBJECTIFICATION":              "p23_objectification",
    "SEXUAL-VIOLENCE":              "p23_sexual_violence",
    "STEREOTYPING-DOMINANCE":       "p23_stereotyping",
}


#Model Labels
MODEL_LABELS = {
    "baseline":       "Early Fusion",
    "gated":          "Gated Fusion",
    "cross_attention":"Cross-Attention",
    "ensemble":       "Ensemble",
}


#DataLoader ---------------------------------------------------------------------------------------
class DataLoader:
    def __init__(self, pred_dir: Path, test_parquet: Path):

        self.pred_dir = Path(pred_dir)

        # Data
        meta = pd.read_parquet(test_parquet)[["id", "lang", "text", "image_file"]]
        meta["id"] = meta["id"].astype(str)
        self.meta = meta.set_index("id")

        # Predictions
        self.preds = {}
        self.available_models = []

        for model in ["baseline", "gated", "cross_attention", "ensemble"]:
            path = self.pred_dir / f"{model}_raw.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                df["id"] = df["id"].astype(str)
                self.preds[model] = df.set_index("id")
                self.available_models.append(model)

        # Gate Values
        self.gates = None
        gates_path = self.pred_dir / "gated_gates.parquet"
        if gates_path.exists():
            g = pd.read_parquet(gates_path)
            g["id"] = g["id"].astype(str)
            self.gates = g.set_index("id")

        #We load our calibrated thresholds
        self.t23 = {}
        if "gated" in self.preds:
            self.t23 = self._calibrate_t23("gated")

        # Build main df
        self.df = self._build_main_df()

    
    
    # Helpers and Functions --------------------------------------------------------------

    #Calculate our calibrated t23 thresholds
    def _calibrate_t23(self, model: str) -> dict:
        df = self.preds[model]
        sexist = df[df["p21"] >= T21]
        thresholds = {}
        for col, freq in TRAIN_FREQ.items():
            thresholds[col] = float(np.percentile(sexist[col], 100 * (1 - freq)))
        return thresholds

    # Task 2.1 Hard (Sexist or not sexist)
    def _hard_21(self, p21: float) -> str:
        return "SEXIST" if p21 >= T21 else "NOT SEXIST"

    # Task 2.2 Hard (Direct / Judgemental)
    def _hard_22(self, p21: float, p_direct: float, p_judg: float) -> str:
        if p21 < T21:
            return "NO"
        return "DIRECT" if p_direct >= p_judg else "JUDGEMENTAL"

    # Task 2.3 Hard (Multilabel)
    def _hard_23(self, p21: float, row: pd.Series, t23: dict) -> list:
        if p21 < T21:
            return ["NO"]
        #Extract labels were the threshold was surprassed 
        active = [label for label, col in CAT_COLS.items() if float(row.get(col, 0)) >= t23.get(col, 0.5)]
        #If not, set the label with the highest value
        if not active:
            best_col = max(CAT_COLS.values(), key=lambda c: float(row.get(c, 0)))
            best_label = [l for l, c in CAT_COLS.items() if c == best_col][0]
            active = [best_label]
        return active

    #Function to merge the data with the predictions with all our models
    def _build_main_df(self) -> pd.DataFrame:
        
        df = self.meta.copy()
        
        for model, preds in self.preds.items():
            short = model[:4]   # base, gate, cros, ense
            df[f"{model}_p21"]  = preds["p21"]
            df[f"{model}_pred21"] = df[f"{model}_p21"].apply(self._hard_21)
            df[f"{model}_p22_direct"] = preds["p22_direct"]
            df[f"{model}_p22_judg"]   = preds["p22_judgemental"]

            for col in CAT_COLS.values():
                df[f"{model}_{col}"] = preds[col]

        # Disagreement score for Task 2.1 (std of p21 across available models)
        p21_cols = [f"{m}_p21" for m in self.available_models]
        if len(p21_cols) > 1:
            df["disagree_score"] = df[p21_cols].std(axis=1)
        else:
            df["disagree_score"] = 0.0

        # Hard pred agreement flag
        pred21_cols = [f"{m}_pred21" for m in self.available_models]
        df["models_agree_21"] = df[pred21_cols].nunique(axis=1) == 1

        return df.reset_index()

    # API Methods ----------------------------------------------------------------
    def get_stats(self) -> dict:
        """Summary statistics for the dashboard."""
        
        
        #General Stats
        stats = {
            "total_memes": len(self.df),
            "lang_distribution": self.df["lang"].value_counts().to_dict(),
            "available_models": self.available_models,
            "model_labels": MODEL_LABELS,
            "agreement_rate_21": float(self.df["models_agree_21"].mean()),
            "models": {}}

        for model in self.available_models:
            p21_col  = f"{model}_p21"
            pred_col = f"{model}_pred21"
            p_d_col  = f"{model}_p22_direct"
            p_j_col  = f"{model}_p22_judg"

            sexist_mask = self.df[pred_col] == "SEXIST"
            stats["models"][model] = {
                "label": MODEL_LABELS.get(model, model),
                "pct_sexist": round(float(sexist_mask.mean()) * 100, 1),
                "pct_direct": round(float(
                    (self.df.loc[sexist_mask, p_d_col] >=
                     self.df.loc[sexist_mask, p_j_col]).mean()
                ) * 100, 1) if sexist_mask.any() else 0,
                "avg_p21": round(float(self.df[p21_col].mean()), 4)}

        # Gate summary
        if self.gates is not None:
            stats["gates_summary"] = {
                "β-mean (Image weight)": round(float(self.gates["gate_beta"].mean()),3),
                "α-mean (EEG weight)": round(float(self.gates["gate_alpha"].mean()),3),
                "λ-mean (ET weight)": round(float(self.gates["gate_lambda"].mean()),3),
            }

        return stats

    #Function to get memes based on lang, prediction (Sexist or not), model, and some text.
    def get_memes(self, page: int, page_size: int,
                  lang: Optional[str] = None,
                  prediction: Optional[str] = None,
                  model: Optional[str] = None,
                  search: Optional[str] = None) -> dict:
        """Filtered, paginated meme list."""
        df = self.df.copy()

        # Filters
        if lang:
            df = df[df["lang"] == lang.lower()]

        if prediction and model and model in self.available_models:
            pred_col = f"{model}_pred21"
            if prediction == "sexist":
                df = df[df[pred_col] == "SEXIST"]
            elif prediction == "not_sexist":
                df = df[df[pred_col] == "NOT SEXIST"]

        if search:
            df = df[df["text"].str.contains(search, case=False, na=False)]

        total = len(df)
        start = (page - 1) * page_size
        page_df = df.iloc[start:start + page_size]

        memes = []
        for _, row in page_df.iterrows():
            #Per Meme
            meme = {
                "id":str(row["id"]),
                "lang":row["lang"],
                "text":str(row["text"])[:300],
                "image_file":str(row["image_file"]),
                "predictions":{}
                }
            #Per that meme we extract the info of that model prediction
            for m in self.available_models:
                meme["predictions"][m] = {
                    "label":MODEL_LABELS.get(m, m),
                    "pred21": row[f"{m}_pred21"],
                    "p21":round(float(row[f"{m}_p21"]), 4),
                }

            meme["models_agree"] = bool(row["models_agree_21"])
            memes.append(meme)

        return {
            "total":total,
            "page":page,
            "page_size":page_size,
            "pages":max(1, -(-total // page_size)),   # ceiling div
            "memes":memes,
        }
    
    #Function to get full detail of ONE meme
    def get_meme_detail(self, meme_id: str) -> Optional[dict]:
        """Full detail for one meme: predictions + gates + categories."""
        row_df = self.df[self.df["id"] == meme_id]

        if row_df.empty:
            return None

        row = row_df.iloc[0]

        detail = {
            "id":meme_id,
            "lang":row["lang"],
            "text":str(row["text"]),
            "image_file":str(row["image_file"]),
            "models_agree_21":bool(row["models_agree_21"]),
            "disagree_score": round(float(row["disagree_score"]), 4),
            "predictions": {},
            "gates": None,}

        # Predictions per model
        for model in self.available_models:
            p21 = float(row[f"{model}_p21"])
            p_d = float(row[f"{model}_p22_direct"])
            p_j = float(row[f"{model}_p22_judg"])
            pred22 = self._hard_22(p21, p_d, p_j)

            cat_probs = {label: round(float(row[f"{model}_{col}"]), 4) for label, col in CAT_COLS.items()}
            pred23 = self._hard_23(p21, row.rename({f"{model}_{col}": col for col in CAT_COLS.values()}),self.t23)

            detail["predictions"][model] = {
                "label":MODEL_LABELS.get(model, model),
                "pred21":self._hard_21(p21),
                "p21":round(p21, 4),
                "pred22":pred22,
                "p22_direct":round(p_d, 4),
                "p22_judg":round(p_j, 4),
                "pred23":pred23,
                "cat_probs":cat_probs}

        # Gate values (Gated model only)
        if self.gates is not None and meme_id in self.gates.index:
            g = self.gates.loc[meme_id]
            detail["gates"] = {
                "β (Image weight)": round(float(g["gate_beta"]),4),
                "α (EEG weight)": round(float(g["gate_alpha"]),4),
                "λ (ET weight)": round(float(g["gate_lambda"]),4)}
            
        return detail

    # Function to calculate and extract disagreements by task
    def get_disagreements(self, page: int, page_size: int, task: str = "2.1") -> dict:
        """Memes sorted by disagreement score."""
        
        df = self.df.copy()

        if task == "2.1":
            # Sort by std of p21 across models (high std = high disagreement)
            df = df.sort_values("disagree_score", ascending=False)
            # Only keep actual disagreements
            df = df[~df["models_agree_21"]]
        else:
            df = df.sort_values("disagree_score", ascending=False)

        total = len(df)
        start = (page - 1) * page_size
        page_df = df.iloc[start:start + page_size]

        memes = []
        for _, row in page_df.iterrows():
            
            meme = {
                "id":str(row["id"]),
                "lang":row["lang"],
                "text":str(row["text"])[:300],
                "image_file":str(row["image_file"]),
                "disagree_score":round(float(row["disagree_score"]), 4),
                "predictions":{}}
            
            for m in self.available_models:
                meme["predictions"][m] = {
                    "label": MODEL_LABELS.get(m, m),
                    "pred21": row[f"{m}_pred21"],
                    "p21": round(float(row[f"{m}_p21"]), 4)}
                
            memes.append(meme)

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": max(1, -(-total // page_size)),
            "memes": memes}
