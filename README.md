# Applying EfficientNetV2 Variants for Face Verification

An advanced face verification system built with PyTorch, comparing EfficientNetV2 variants (Small, Medium, Large) using Additive Angular Margin Loss (ArcFace) and Online Hard Pair Mining. Evaluated on the standard LFW 6,000-pair benchmark.

## Team Members (Group 8)

* **Le Viet Cuong** - Project Manager, Eval & Pipeline Setup
* **Hoang Quoc Huy** - Data Engineer (MTCNN & Hard Miner)
* **Dinh Ha Hai** - ML Engineer (Baseline Siamese Network)
* **Nguyen Ngoc Linh** - ML Engineer (ArcFace & Demo)

## Key Features

* **State-of-the-Art Backbone:** Comparative study of EfficientNetV2 S/M/L architectures.
* **Advanced Metric Learning:** ArcFace ($m=0.5, s=64$) with tighter cluster margins.
* **Online Hard Pair Mining (OHPM):** Batch-level hard positive/negative mining during ArcFace training.
* **Rigorous Evaluation:** FAR, FRR, EER, ROC plots, and per-pair latency metrics.

---

## Project Structure

```
configs/                  # 6 YAML training configs (baseline + arcface, S/M/L)
data/
  raw/{lfw,calfw,cplfw}/   # Raw datasets + pairs.csv
  processed/{lfw,calfw,cplfw}/{S,M,L}/  # Preprocessed crops
demo/verify.py            # Live webcam verification demo
scripts/                  # CLI entry points
src/
  data/                   # Datasets, MTCNN preprocessing, hard pair miner
  models/                 # Backbone, Siamese network, losses
  training/               # Baseline & ArcFace training scripts
  evaluation/             # Metrics, LFW evaluation, error analysis
  utils/                  # Paths and config loading
outputs/
  checkpoints/            # Saved model weights
  metrics/                # comparison_table.csv, ROC/FAR-FRR plots
  error_cases/            # False accept/reject visualizations
  results/                # Preprocessing logs
```

---

## Environment Setup

**1. Create a Virtual Environment (Python 3.11+)**

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

**2. Install PyTorch**

GPU (CUDA 11.8):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

CPU only:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**3. Install Project Dependencies**

```bash
pip install -r requirements.txt
```

Or use the automated setup script:

```bash
# Mac/Linux
bash setup.sh

# Windows (PowerShell)
.\setup.ps1
```

---

## Dataset Setup

Three datasets are used with **evaluation pairs only**:

| Dataset | Role | Eval pairs | Challenge |
|---|---|---|---|
| **CALFW** | Train + Val (merged with CPLFW) | 6,000 | Cross-age |
| **CPLFW** | Train + Val (merged with CALFW) | 6,000 | Cross-pose |
| **LFW** | Test only (held-out) | 6,000 | Standard benchmark |

**Folder layout:**

```
data/
  raw/
    lfw/pairs.csv
    calfw/calfw/pairs.csv
    cplfw/cplfw/pairs.csv
  processed/
    lfw/{S,M,L}/{identity}/*.jpg
    calfw/{S,M,L}/aligned images/*.jpg
    cplfw/{S,M,L}/aligned images/*.jpg
```

**Train/Val split:** CALFW + CPLFW eval pairs are merged (12,000 total), then split **80% train / 20% val** (`split_seed: 42` in configs).

**Test:** LFW eval pairs only — never used during training.

If you need to re-run MTCNN preprocessing for LFW:

```bash
python scripts/mtcnn_preprocess.py
```

---

## How to Run the Pipeline

**Step 1: Train Baseline Models (Contrastive Loss)**

```bash
python scripts/train_baseline.py --config configs/baseline_s.yaml
python scripts/train_baseline.py --config configs/baseline_m.yaml
python scripts/train_baseline.py --config configs/baseline_l.yaml
```

**Step 2: Train Advanced Models (ArcFace + Hard Mining)**

```bash
python scripts/train_arcface.py --config configs/arcface_s.yaml
python scripts/train_arcface.py --config configs/arcface_m.yaml
python scripts/train_arcface.py --config configs/arcface_l.yaml
```

**Step 3: Evaluate on LFW Test Set**

```bash
python scripts/evaluate_lfw.py
```

Runs all 6 models on **LFW eval pairs only** (held-out test set).

**Step 4: Visual Error Analysis**

```bash
python scripts/error_analysis.py
```

Output: side-by-side false accept/reject cases in `outputs/error_cases/`.

**Step 5: Live Demo**

```bash
python demo/verify.py
```

Enroll a face from webcam, capture a test face, and see MATCH/REJECT with similarity score and threshold.

---

## Training Hyperparameters

All configs follow the shared project spec:

| Parameter | Value |
|---|---|
| Epochs | 30 |
| Learning rate | 1e-4 |
| Optimizer | AdamW (weight_decay=0.4) |
| Dropout | 0.3 (before embedding head) |
| Early stopping | 5 epochs (on validation EER) |
| Layer unfreeze | 30% |
| Batch sizes | S=16, M=8, L=4 |
| ArcFace | s=64, m=0.5 |

Checkpoints are saved to `outputs/checkpoints/` when validation EER improves.

---

## Expected Outputs

* `outputs/checkpoints/baseline_{s,m,l}.pt` — contrastive loss models
* `outputs/checkpoints/arcface_{s,m,l}.pt` — ArcFace + hard mining models
* `outputs/metrics/comparison_table.csv` — side-by-side EER/FAR/latency table
* `outputs/metrics/*_roc.png` — ROC curves per model
* `outputs/error_cases/` — false accept/reject visualizations

**Target:** EER < 5% on the LFW 6,000-pair benchmark.
