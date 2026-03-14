# Urdu → Roman Urdu Transliterator

A sequence-to-sequence neural transliterator that converts **Urdu script (نستعلیق)** into **Roman Urdu** using a BiLSTM encoder and LSTM decoder, trained on Urdu poetry (ghazals).

🚀 **Live Demo:** [Streamlit App](https://share.streamlit.io/aurangzaibawan/urduto-roman-transliterator/main/app.py)  
📝 **Write-up:** [Medium Article](https://medium.com/@aurangzaibshehzadawan/learning-through-building-urdu-roman-urdu-translator-with-bilstm-e78249ac31c6)

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [Key Learnings](#key-learnings)

---

## Overview

Roman Urdu has no standardized spelling — people write it however they like in casual digital communication. This makes it a genuinely hard, low-resource NLP task. This project tackles that by training a character-level Seq2Seq model on Urdu poetry (ghazals), where the language is dense with metaphor and the dataset is small.

**Example:**  
Input: `محبت`  
Output: `mohabbat`

---

## Dataset

- **Source:** [`urdu_ghazals_rekhta`](https://github.com/amir9ume/urdu_ghazals_rekhta) — Urdu poetry with paired transliterations
- **Preprocessing:**
  - Normalized Urdu characters
  - Removed poetic symbols (e.g., `|`)
  - Character-level tokenization (chosen over word-level due to inconsistent Roman Urdu spelling)
- **Split:** 50% train / 25% validation / 25% test

---

## Model Architecture

A **Seq2Seq** model implemented in PyTorch:

| Component | Details |
|-----------|---------|
| Encoder | 2-layer Bidirectional LSTM (BiLSTM) |
| Decoder | 4-layer LSTM |
| Embedding size | 128–512 (experimented) |
| Hidden size | 256 / 512 |
| Dropout | 0.1–0.5 |
| Training technique | Teacher forcing |

The encoder reads Urdu characters in both directions; the decoder generates Roman Urdu character by character.

---

## Training Details

- **Framework:** PyTorch
- **Optimizer:** Adam with tuned learning rate
- **Key challenge:** Overfitting on the small poetic dataset
- **Mitigations:** Dropout, lower learning rate, larger embeddings

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **BLEU** | Translation quality |
| **Perplexity** | Model uncertainty |
| **CER** | Character Error Rate via Levenshtein distance |

---

## Project Structure

```
urdu_roman_transliterator/
│
├── assign1/              # Core model code
│   ├── *.py              # Training, model, preprocessing scripts
│   └── ...
│
└── README.md
```

---

## Setup & Usage

### Prerequisites

```bash
pip install torch streamlit
```

### Clone the repo

```bash
git clone https://github.com/Aurangzaib-Awan/urdu_roman_transliterator.git
cd urdu_roman_transliterator
```

### Run the Streamlit App

```bash
streamlit run app.py
```

Or use the [live demo](https://share.streamlit.io/aurangzaibawan/urduto-roman-transliterator/main/app.py) directly.

---

## Key Learnings

- **Teacher forcing** helps during training but can hurt generalization at test time — requires careful scheduling
- **Low-resource poetic data** forces rigorous thinking about noise, normalization, and data quality
- **Character-level tokenization** works better than word-level for Roman Urdu due to spelling inconsistencies
- Deployment matters as much as modeling — wrapping the model in Streamlit made it immediately usable

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

---

## Author

**Aurangzaib Shehzad Awan**  
[GitHub](https://github.com/Aurangzaib-Awan) · [Medium](https://medium.com/@aurangzaibshehzadawan)
