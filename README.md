# Multimodal EXIST  
### Text + Image + Neurophysiological Multimodality for Sexism Detection in Memes
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-demo-009688?style=flat&logo=fastapi&logoColor=white)


# Overview

**Multimodal EXIST** is a multimodal and multitask learning project focused on detecting and categorizing sexist content in memes. The project combines **text and image** content with **neurophysiological signals** such as Eye Tracking, EEG, and Heart Rate.

The goal is to build models that not only understand what is shown and written in a meme, but also incorporate how humans cognitively and physiologically react to that content.

This project explores whether if physiological information can improve multimodal classification beyond standard text-image models.

---

## Dataset

**EXIST 2026 Memes Dataset**: ~5,000 labeled memes in English and Spanish.

| Split | Memes |
|---|---|
| Train | 3,984 |
| Test | 1,053 |

Each meme includes meme image, OCR-extracted text, human annotations (6 annotators per meme), Eye Tracking signals, EEG band-power features (16 channels × 5 bands), and Heart Rate measurements.

**Tasks:**

| Task | Type | Labels |
|---|---|---|
| 2.1 — Sexism ID | Binary | YES / NO |
| 2.2 — Source Intention | Multiclass | DIRECT / JUDGEMENTAL |
| 2.3 — Categorization | Multilabel | Ideological · Stereotyping · Objectification · Sexual Violence · Misogyny |

---

## Modalities

| Modality | Description |
|---|---|
|*Text*|The OCR-extracted text from the meme is encoded using transformer-based language models.|
|*Image*|The meme image is represented using visual encoders such as vision transformers or vision-language encoders.|
|*Eye Tracking*|Eye Tracking features capture visual attention and cognitive processing, including reaction time, fixation behavior, and saccades.|
|*EEG*|EEG features capture neurophysiological activity through frequency-band representations (Alpha, Beta, Theta, Delta, and Gamma).|
|*Heart Rate*|Heart Rate is included as an auxiliary physiological signal and can be evaluated through ablation experiments.|

---
## Models

### 1. Early Fusion (Baseline)
Text (mDeBERTa-v3) + Image (ALIGN) + Physio (EEG+ET MLP) concatenated → shared classifier heads for all 3 tasks.

### 2. Gated Fusion
Same encoders, but the fusion learns scalar gates per modality conditioned on the text representation:

```
z = text + β·image + α·EEG + λ·ET
```

Gates are sigmoid outputs with temperature 0.3. Produces interpretable, per-meme modality weights.

### 3. Cross-Attention Gated
Bidirectional cross-attention between text↔image and text↔physio before gated fusion. Captures inter-modal interactions at the representation level before weighting.

All models are trained multi-task (Tasks 2.1, 2.2, 2.3 simultaneously) with masked loss and 5-fold stratified CV.

---

## Evaluation Metrics

3-fold cross-validation on the test set (1,053 memes, EN + ES).

- Task 2.1: binary sexism detection (Macro F1)
- Task 2.2: source intention — DIRECT vs JUDGEMENTAL (Macro F1)
- Task 2.3: fine-grained category detection — multilabel (Macro F1)
- AUC was also calculated for a secondary metric.

## Key Findings

- **Modality hierarchy:** The Gated Fusion model consistently learned `β(Image) ≈ 0.98 > α(EEG) ≈ 0.78 > λ(ET) ≈ 0.10` — image dominates, EEG adds signal, eye-tracking is suppressed.
- **EEG improves fine-grained categorization:** Physiological signals most benefit Task 2.3 (category detection), where content features alone are weakest — consistent with prior work showing EEG AUC of 0.717 rivaling text-only models.
- **OCR text-image redundancy:** In memes, the text input is OCR-extracted from the image itself, creating cross-modal redundancy. The Cross-Attention model reveals this by producing unstable image gates (β: 0.07–1.00 across folds), while the Gated model remains stable.
- **Interpretable fusion:** Per-meme gate values expose *which brain regions the model relies on* for each meme — a novel interpretability angle for multimodal hate speech detection.

---

## Demo API

A FastAPI + vanilla JS demo for exploring predictions and gate visualizations.

```bash
cd inference/api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000`

**Pages:**
- **Overview** — global stats, model comparison, gate summary
- **Explorer** — browse all 1,053 test memes with per-model predictions
- **Gates** — distribution of β/α/λ gate values, sortable by modality
- **Disagree** — memes where models diverge, ranked by uncertainty
- **Train Dataset** — human annotation scores across 3,984 training memes

---