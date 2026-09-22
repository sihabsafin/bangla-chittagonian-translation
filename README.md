<div align="center">

# 🗣️ Bangla → Chittagonian Dialect Translation
### Parameter-Efficient Fine-Tuning of Small Language Models with QLoRA

*A B.Sc. thesis project on low-resource dialect translation for Bangladesh's regional languages*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Transformers-yellow?style=flat-square)](https://huggingface.co/)
[![QLoRA](https://img.shields.io/badge/Fine--Tuning-QLoRA-8A2BE2?style=flat-square)](https://arxiv.org/abs/2305.14314)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](#license)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)]()

[Overview](#-overview) •
[Models](#-models) •
[Dataset](#-dataset) •
[Methodology](#-methodology) •
[Results](#-results) •
[Setup](#-installation) •
[Usage](#-usage)

</div>

---

## 📖 Overview

**Chittagonian** is a widely spoken regional language variety in southeastern Bangladesh, yet it remains a low-resource dialect with almost no dedicated NLP tooling. This project investigates whether **small, instruction-tuned language models** — fine-tuned with **QLoRA** — can learn accurate Bangla → Chittagonian translation from a curated parallel corpus, under realistic, constrained compute (a single Colab T4 GPU).

The work covers the full pipeline: dataset construction and native-speaker validation, preprocessing, prompt design, multi-model QLoRA fine-tuning, and evaluation using complementary word-level and character-level MT metrics.

### Research Questions

| # | Question |
|---|----------|
| 1 | How effectively can parameter-efficient adaptation specialize instruction-tuned Small Language Models for Bangla-to-Chittagonian translation? |
| 2 | How do Qwen2.5-3B-Instruct, Gemma-2B-IT, and Llama-3.2-3B-Instruct differ under a unified QLoRA framework? |
| 3 | How do word-level and character-level metrics (BLEU vs. chrF++) reflect model performance for low-resource Chittagonian translation? |

---

## 🖼️ Thesis Defense

<div align="center">
<img src="sihabsafin.png" alt="Sihabul Islam Safin defending his B.Sc. thesis" width="520"/>

<sub>Presenting the final thesis at BGC Trust University Bangladesh</sub>
</div>

---

## 🧠 Models

Three instruction-tuned small language models were fine-tuned and evaluated under an identical QLoRA protocol for a controlled comparison:

| Model | Parameters | Source |
|---|---|---|
| **Qwen2.5-3B-Instruct** | 3B | Alibaba / Qwen |
| **Gemma-2B-IT** | 2B | Google |
| **Llama-3.2-3B-Instruct** | 3B | Meta |

---

## 📊 Dataset

A curated **English–Bangla–Chittagonian** parallel corpus was built specifically for this thesis from publicly available online linguistic resources, then refined using native-speaker input — no adequate public dataset existed for this exact setting.

| Property | Value |
|---|---|
| Raw records | 7,663 |
| Final cleaned records | **7,630** |
| Language fields | English, Standard Bangla, Chittagonian (English retained as auxiliary reference; **not** used as model input) |
| Pool split | 80% train (6,104) / 10% validation (763) / 10% test (763) — fixed random seed 42 |
| Examples actually used for fine-tuning | **800 train / 100 validation** (low-resource experimental setting) |
| Final evaluation | Full held-out **763-example** test set for all three models |
| Format | Instruction-style Bangla→Chittagonian prompt pairs |
| Leakage check | No exact-duplicate Bangla or Chittagonian sentences found across train/val/test splits |

All target-side translations were reviewed and refined with native-speaker input before use, with particular attention to distinguishing natural Chittagonian expression from literal, overly Standard-Bangla-influenced phrasing.

---

## ⚙️ Methodology

```
Data Collection → Native-Speaker Validation → Cleaning & Preprocessing → Train/Val/Test Split (80/10/10)
      → Subsample for Low-Resource Setting (800/100/763) → Prompt Engineering
      → QLoRA Fine-Tuning (×3 models) → Inference → Evaluation (BLEU / chrF++) → Qualitative Error Analysis
```

1. **Dataset Collection** — Gathered English–Bangla–Chittagonian sentence triples from publicly available online linguistic resources.
2. **Native-Speaker Validation** — Refined Chittagonian translations for naturalness and dialectal accuracy.
3. **Cleaning & Preprocessing** — Removed blank/incomplete records, stripped whitespace, applied NFC Unicode normalization, dropped entries under 3 characters, removed duplicate/identical Bangla–Chittagonian pairs.
4. **Splitting** — 80/10/10 train/validation/test pools (seed 42); 800/100 examples subsampled from train/validation for the low-resource fine-tuning setting, with the full 763-example test set reserved for evaluation.
5. **Prompt Engineering** — Formulated translation as an instruction-following task: `Translate the following Bangla sentence into Chittagonian.` The English field is deliberately excluded from prompts.
6. **QLoRA Fine-Tuning** — 4-bit NF4 quantization with LoRA adapters (r=16, α=32, dropout 0.05) on the q/k/v/o attention projections; base weights frozen, only adapter parameters trained.
7. **Generation** — Produced Chittagonian translations for all 763 held-out test examples, per model.
8. **Evaluation** — Scored with BLEU and chrF++, followed by qualitative error analysis across seven error categories.

### Why QLoRA?

QLoRA enables fine-tuning of billion-parameter models on a single consumer-grade GPU by combining 4-bit NF4 quantization with low-rank adapters — freezing the base model and training only a small set of adapter weights. This made it possible to fine-tune three separate 2–3B models end-to-end within Colab's free-tier constraints, using just 800 training examples, without sacrificing meaningful translation quality.

---

## 🏆 Results

All three models were trained and evaluated on the identical 800/100 train/validation subsets and the full 763-example held-out test set, using the same QLoRA configuration (rank 16, alpha 32, dropout 0.05, targeting q/k/v/o projections).

| Model | BLEU | chrF++ |
|---|:---:|:---:|
| **Qwen2.5-3B-Instruct** 🥇 | **0.4179** | 12.7246 |
| Gemma-2B-IT | 0.2712 | 14.2280 |
| **Llama-3.2-3B-Instruct** 🥇 | 0.3755 | **16.8909** |

<div align="center">
<img src="outputs/comparison/model_comparison.png" alt="Model comparison bar chart across BLEU and chrF++ metrics" width="700"/>
</div>

**Key observations:**
- **Qwen2.5-3B-Instruct** scored highest on **BLEU** (0.4179), indicating the strongest word- and phrase-level n-gram agreement with reference translations.
- **Llama-3.2-3B-Instruct** scored highest on **chrF++** (16.8909), a character-level metric more forgiving of morphological and orthographic variation — relevant for a dialect without standardized spelling.
- **No single model wins on both metrics.** Word-level and character-level evaluation surface different strengths: Qwen's outputs match reference wording more exactly, while Llama's outputs are closer at the character/morphology level even where exact word matches diverge.
- **ROUGE was dropped from the final evaluation.** It is designed for summarization-style overlap and produced uninformative (near-zero) scores on short, single-sentence, high-lexical-divergence dialect translations; BLEU and chrF++ were adopted as the primary metrics instead.
- A **qualitative error analysis** across seven categories — lexical mismatch, word-order error, morphological variation, spelling variation, untranslated words, Standard Bangla influence, and semantic error — complements the automatic scores and shows that translation quality can't be judged from exact lexical overlap alone.

### Fine-Tuning Configuration

| Parameter | Setting |
|---|:---:|
| Fine-tuning method | QLoRA |
| Quantization | 4-bit NF4 |
| LoRA rank (r) | 16 |
| LoRA alpha (α) | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Max sequence length | 256 |
| Training examples | 800 |
| Validation examples | 100 |
| Test examples | 763 (full held-out set) |
| Epochs | 2 |
| Batch size | 1 |
| Gradient accumulation | 8 |
| Learning rate | 2 × 10⁻⁴ |
| Warmup ratio | 0.03 |

> The same training/validation subsets, prompt structure, hyperparameters, and evaluation procedure were applied identically across all three models to keep the comparison controlled.

---

## 🗂️ Repository Structure

```text
Bangla-NLP/
├── data/
│   ├── raw/                # Original, unprocessed collected data
│   ├── interim/             # Partially cleaned intermediate data
│   ├── processed/           # Final, model-ready datasets
│   └── external/            # Any third-party reference data
├── outputs/
│   ├── evaluation/          # Raw evaluation results (BLEU, chrF++)
│   ├── figures/              # Plots and visualizations
│   ├── tables/                # Summary result tables
│   ├── reports/               # Thesis-related reports
│   └── models/                 # Saved QLoRA adapter checkpoints
├── src/
│   ├── data/                 # Dataset loading utilities
│   ├── eda/                   # Exploratory data analysis
│   ├── preprocessing/         # Cleaning & normalization scripts
│   ├── features/               # Prompt formatting / feature engineering
│   ├── models/                  # Model + QLoRA configuration
│   ├── training/                 # Fine-tuning pipelines
│   ├── evaluation/                # Metric computation scripts
│   ├── explainability/             # Analysis of model outputs
│   └── utils/                       # Shared helper functions
├── notebooks/                # Colab / Jupyter notebooks
├── assets/
│   └── model_comparison.png  # Evaluation results chart
├── sihabsafin.png             # Thesis defense photo
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

<div align="left">

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/-Transformers%204.46.0-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![PEFT](https://img.shields.io/badge/-PEFT%200.12.0-8A2BE2?style=flat-square)
![TRL](https://img.shields.io/badge/-TRL%200.9.6-FF6F00?style=flat-square)
![Accelerate](https://img.shields.io/badge/-Accelerate%200.34.0-666666?style=flat-square)
![BitsAndBytes](https://img.shields.io/badge/-BitsAndBytes%200.45.5-333333?style=flat-square)
![Datasets](https://img.shields.io/badge/-Datasets%202.21.0-FFD21E?style=flat-square)
![SentencePiece](https://img.shields.io/badge/-SentencePiece%200.2.0-555555?style=flat-square)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat-square)
![Google Colab](https://img.shields.io/badge/-Google%20Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)

</div>

Experiments were run in a GPU-enabled Google Colaboratory environment on a single **NVIDIA T4 GPU**, with fixed software versions used throughout for reproducibility.

---

## 🚀 Installation

```bash
git clone https://github.com/sihabsafin/bangla-chittagonian-translation
cd Bangla-NLP
pip install -r requirements.txt
```

> Running on Google Colab? Open the notebooks in `notebooks/` and make sure a GPU runtime is enabled (`Runtime → Change runtime type → GPU`).

---

## ▶️ Usage

**1. Prepare the dataset**
Ensure the cleaned/formatted dataset is placed in `data/processed/`.

**2. Preprocess**
```bash
python src/preprocessing/clean_data.py
```

**3. Format prompts**
```bash
python src/features/build_prompts.py
```

**4. Fine-tune with QLoRA**
```bash
python src/training/train_qlora.py --model qwen2.5-3b
```

**5. Evaluate**
```bash
python src/evaluation/evaluate.py --metrics bleu chrf
```

---

## 🔭 Future Work

- Expand the parallel corpus with more domains, regional variation, and speaker demographics
- Incorporate human evaluation by native Chittagonian speakers alongside automatic metrics
- Benchmark additional/larger small language models and adapter configurations
- Move from sentence-level to context-aware or document-level translation
- Package the translation model as a web or mobile application

---

## 📌 Notes

This is a research project focused on low-resource dialect translation, built under limited computational resources (single Colab T4 GPU) and a deliberately small 800-example fine-tuning subset to study the low-resource regime. Results may vary depending on hardware, dataset version, and training hyperparameters, and should be interpreted as a comparison among the three selected models rather than a universal ranking of all small language models. See the accompanying thesis paper for full limitations and ethical considerations.

---

## 🙏 Acknowledgements

Special thanks to my thesis supervisor **Ferdous Ara** for guidance throughout this project, as well as everyone who contributed sentence pairs, native-speaker validation, feedback, and support along the way.

---

## 📄 License

This project is released under the MIT License — see [`LICENSE`](LICENSE) for details.

---

## 📬 Contact

<div align="center">

**Sihabul Islam Safin**
Final-Year CSE Student, BGC Trust University Bangladesh · Freelance Generative AI Engineer

[![GitHub](https://img.shields.io/badge/GitHub-sihabsafin-181717?style=flat-square&logo=github)](https://github.com/sihabsafin/bangla-chittagonian-translation)
[![Email](https://img.shields.io/badge/Email-sihabulislamsafin%40bgctub.ac.bd-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:sihabulislamsafin@bgctub.ac.bd)

</div>
