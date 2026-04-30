# Multimodal EXIST  
### Text + Image + Neurophysiological Multimodality for Sexism Detection in Memes

## Overview

**Multimodal EXIST** is a multimodal and multitask learning project focused on detecting and categorizing sexist content in memes. The project combines **text and image** content with **neurophysiological signals** such as Eye Tracking, EEG, and Heart Rate.

The goal is to build models that not only understand what is shown and written in a meme, but also incorporate how humans cognitively and physiologically react to that content.

This project explores whether if physiological information can improve multimodal classification beyond standard text-image models.

---

## Dataset

The project is based on the **EXIST 2026 Memes Dataset**, which contains approximately **3,000 labeled memes** in English and Spanish.

Each meme includes:

- Meme image
- OCR-extracted text
- Human annotations
- Eye Tracking signals
- EEG band-power features
- Heart Rate measurements
- Annotator-level metadata

---

## Prediction Tasks

The project focuses on three main tasks:

### Task 2.1 — Sexism Identification

Binary classification task:

- `0`: Non-sexist
- `1`: Sexist

### Task 2.2 — Source Intention Detection

Binary classification task over sexist memes:

- `DIRECT`
- `JUDGEMENTAL`

### Task 2.3 — Sexism Categorization

Multilabel classification task identifying the type(s) of sexism present in a meme:

- Ideological inequality
- Stereotyping dominance
- Objectification
- Sexual violence
- Misogyny / non-sexual violence

---

## Modalities

The proposed models integrate the following modalities:

### Text

The OCR-extracted text from the meme is encoded using transformer-based language models.

### Image

The meme image is represented using visual encoders such as vision transformers or vision-language encoders.

### Eye Tracking

Eye Tracking features capture visual attention and cognitive processing, including reaction time, fixation behavior, and saccades.

### EEG

EEG features capture neurophysiological activity through frequency-band representations such as Alpha, Beta, Theta, Delta, and Gamma.

### Heart Rate

Heart Rate is included as an auxiliary physiological signal and can be evaluated through ablation experiments.

---

## Modeling Approach

This project evaluates three progressively more expressive multimodal architectures.

### 1. Early Fusion

A baseline multimodal model where text, image, EEG, ET, and HR representations are concatenated into a single feature vector before classification.

### 2. Gated Fusion

A more adaptive architecture where the model learns how much weight to assign to each modality.

The fusion representation follows the intuition:

$$
z = Text + \alpha * Image + \beta * ET + \lambda * EEG
$$

### 3. Cross-Attention + Gated Fusion

A more advanced architecture where cross-attention is used to model interactions between modalities before applying gated fusion.  This allows the model to learn relationships between meme content and physiological responses, while still controlling the final contribution of each modality through adaptive gates.

---

## Evaluation Metrics

The main evaluation metrics are:

- Macro F1-Score
    - Binary F1 for Task 2.1
    - Binary F1 for Task 2.2
    - Multilabel F1 for Task 2.3
- AUC


