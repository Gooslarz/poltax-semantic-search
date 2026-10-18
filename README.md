# PolTax Semantic Search: Hybrid IR & LoRA Domain Adaptation for Polish Tax Law

A modular Information Retrieval (IR) research prototype built with **PyTorch**, **Hugging Face Transformers**, and **Scikit-learn** for semantic search over Polish individual tax interpretations (*interpretacje indywidualne KIS* across PIT, VAT, and CIT).

## Key Engineering & Research Features
* **Attention-Masked Neural Pooling:** Computes L2-normalized sentence representations from transformer token embeddings (`last_hidden_state`) using `attention_mask` tensor broadcasting to eliminate `<pad>` token dilution.
* **Polish & Multilingual Encoders (Hugging Face):** Benchmarks `intfloat/multilingual-e5-small` (384-dim) and the native Polish retrieval encoder `sdadas/mmlw-retrieval-roberta-base` (768-dim).
* **Low-Rank Contrastive Domain Adaptation:** Implements a custom PyTorch `nn.Module` (`TaxDomainAdapter`) using a LoRA-style bottleneck projection (`768 -> 16 -> 768`, reducing trainable parameters by **95.8%** from 590,592 to 24,576 to prevent overfitting on small legal queries).
* **Listwise InfoNCE Training:** Trains the adapter on a separate 16-query training set using temperature-scaled negatives and best-checkpoint restoration (`tax_adapter.pt`), evaluating strictly on unseen test queries.
* **Automated Functional Test Suite:** Uses `pytest` to verify tensor shapes, unit hypersphere L2 normalization, score bounds, and Polish legal notation handling (`art.`, `§`, `PKWiU`).

## Benchmark Results (Unseen Test Queries)

| Method / Architecture | Recall@1 | Recall@3 | MRR |
| :--- | :--- | :--- | ---: |
| 1. Scikit-learn TF-IDF (Lexical Baseline, alpha=0.0) | 75.0% | 87.5% | 0.823 |
| 2. PyTorch + `multilingual-e5-small` (Dense, alpha=1.0) | 87.5% | 87.5% | 0.906 |
| 3. PyTorch + Polish `MMLW-RoBERTa` (Frozen Dense, alpha=1.0) | 75.0% | 100.0% | 0.833 |
| **4. PyTorch + Polish `MMLW` + Trained LoRA Adapter (InfoNCE)** | **100.0%** | **100.0%** | **1.000** |
| 5. Hybrid: Polish `MMLW` + TF-IDF (alpha=0.65) | 87.5% | 100.0% | 0.917 |
| 6. Hybrid: `multilingual-e5` + TF-IDF alpha=0.65) | 87.5% | 87.5% | 0.906 |

### Per-Question Rank X-Ray (`1` = Ranked #1)

| Q# | Query Category | M1 (TF-IDF) | M2 (E5) | M3 (MMLW) | **M4 (MMLW + LoRA)** | M5 (Hybrid MMLW) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Q1** | Standard Semantic (B2B Glasses Expense) | 1 | 1 | 1 | **1** | 1 |
| **Q2** | Colloquial Paraphrase (Algorithm IP Tax) | 1 | 1 | 3 | **1** | 1 |
| **Q3** | Standard Semantic (Optical VAT 8%) | 1 | 1 | 1 | **1** | 1 |
| **Q4** | Statute / Entity Trap (IT + 150k Car Limit) | 3 | 4 | 3 | **1** | 3 |
| **Q5** | Colloquial Paraphrase (EV 225k Limit) | 1 | 1 | 1 | **1** | 1 |
| **Q6** | Exact Legal Classification (PKWiU 62.01) | 1 | 1 | 1 | **1** | 1 |
| **Q7** | Colloquial Paraphrase (Home Office Utilities) | 4 | 1 | 1 | **1** | 1 |
| **Q8** | Standard Semantic (US Export Art. 28b VAT) | 1 | 1 | 1 | **1** | 1 |

## Quickstart
```bash
pip install -r requirements.txt
python train_adapter.py
python evaluate.py
``` 

For testing:
```bash
pytest test_engine.py -v
```
