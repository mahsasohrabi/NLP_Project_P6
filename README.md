# P6: The Apprentice Model
## Financial Sentiment Knowledge Distillation

**NLP 2025/26 · Università degli Studi di Milano · Prof. Alfio Ferrara**

---

## Domain & Dataset Choice

### Why Financial Sentiment Analysis?

| Criterion | Justification |
|-----------|--------------|
| **Clean task** | 3-class classification (neg / neu / pos) — clear metrics |
| **Measurable** | Macro-F1 captures class imbalance honestly |
| **Business-relevant** | Directly applicable to trading, risk management, ESG |
| **Well-benchmarked** | Strong prior results (FinBERT ~0.87 F1) to compare against |
| **LLM teacher is meaningful** | Claude produces calibrated confidence on ambiguous financial language |

### Dataset: FinancialPhraseBank (`sentences_allagree`)

- **Source:** Malo et al. (2014), HuggingFace Hub: `financial_phrasebank`
- **Size:** 2,264 sentences (strictest agreement split)
- **Labels:** negative (9%), neutral (59%), positive (32%)
- **Split:** 70/15/15 train/val/test (stratified)

---

## Project Structure

```
apprentice-model/
├── main.py                        ← orchestrates full pipeline
├── requirements.txt
├── notebooks/
│   └── demo.ipynb                 ← step-by-step walkthrough
└── src/
    ├── data/
    │   └── loader.py              ← FinancialPhraseBank loading & splitting
    ├── models/
    │   ├── teacher.py             ← Claude API annotation + mock teacher
    │   └── student.py             ← DistilBERT training (baseline + distillation)
    ├── evaluation/
    │   └── metrics.py             ← accuracy, F1, confusion matrix, speed
    └── visualization/
        └── plots.py               ← training curves, confusion matrices, comparison
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run with mock teacher (no API key needed)
python main.py --mock

# 3. Run with real Claude teacher
python main.py --api-key sk-ant-...

# 4. Interactive demo notebook
jupyter notebook notebooks/demo.ipynb
```

---

## Distillation Method

**Response-based distillation** (Hinton et al., 2015):

```
L = α · L_KD(student/T ∥ teacher/T) · T²  +  (1 − α) · L_CE(student, y_hard)
```

- **T = 4.0** — temperature softens probability distributions
- **α = 0.7** — 70% weight on KD loss, 30% on hard-label CE loss
- **Teacher:** Claude generates P(negative, neutral, positive) per sentence
- **Student:** DistilBERT-base-uncased (66M params, 40% smaller than BERT-base)

---

## Research Question

> *Does distillation from Claude improve financial sentiment classification
> in a compact DistilBERT student compared to standard fine-tuning?*

**Hypotheses:**
- H1: Distilled student ≥ baseline on Macro-F1
- H2: Gains strongest on minority class (negative) where ambiguity is highest  
- H3: Temperature T=4 outperforms T=1 and T=8
- H4: Higher α benefits ambiguous/borderline sentences most

---

## Key Files to Read for Paper

| Section | File |
|---------|------|
| Introduction | This README + `src/data/loader.py` docstring |
| Research Question & Methodology | `src/models/student.py` (loss function) + `src/models/teacher.py` |
| Experimental Results | `outputs/evaluation/comparison_report.json` + `outputs/figures/` |
| Concluding Remarks | Interpret `outputs/evaluation/comparison_report.json` |

---

## References

- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network.
- Jiao, X., et al. (2020). TinyBERT: Distilling BERT for Natural Language Understanding. *EMNLP Findings*.
- Malo, P., et al. (2014). Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts.
- Lewis, P., et al. (2020). FinBERT: A Pretrained Language Model for Financial Communications.

---

## AI Usage Disclaimer

Parts of this project structure and documentation were developed with assistance from Claude (Anthropic).
All code has been reviewed and validated. The student takes full responsibility for the final content,
experimental design, and academic integrity of the submitted work.
