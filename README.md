# Applying EfficientNetV2 Variants for Face Verification

An advanced face verification system built with PyTorch, comparing EfficientNetV2 variants (Small, Medium, Large) using Additive Angular Margin Loss (ArcFace) and Online Hard Pair Mining. Evaluated on the standard LFW 6,000-pair benchmark.

## Team Members (Group 8)

* **Le Viet Cuong** - Project Manager, Eval & Pipeline Setup
* **Hoang Quoc Huy** - Data Engineer (MTCNN & Hard Miner)
* **Dinh Ha Hai** - ML Engineer (Baseline Siamese Network)
* **Nguyen Ngoc Linh** - ML Engineer (ArcFace & Demo)

## Data Protocol

| Split | Dataset | Role | Augmentation |
|---|---|---|---|
| **Train** | CASIA-WebFace | ~10k identities, metric-learning pre-training | None |
| **Val** | CALFW + CPLFW | 12,000 official eval pairs combined | None |
| **Test** | LFW | 6,000 official eval pairs (held-out) | Horizontal-flip TTA only |

```
archive/casia-webface/          # MXNet RecordIO (train.rec, train.idx, train.lst)
data/raw/lfw/                   # LFW deepfunneled + pairs.csv
data/raw/calfw/calfw/           # CALFW images + pairs.csv
data/raw/cplfw/cplfw/           # CPLFW images + pairs.csv
data/processed/
  casia-webface/{S,M,L}/        # Extracted training crops
  lfw/{S,M,L}/                  # MTCNN-aligned test crops
  calfw/{S,M,L}/                # MTCNN-aligned val crops
  cplfw/{S,M,L}/                # MTCNN-aligned val crops
```

---

## Environment Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Or run `.\setup.ps1` (then `pip install mxnet` if not already installed).

---

## Step-by-Step Pipeline

### Step 0 — Place datasets

1. **CASIA-WebFace** is already in `archive/casia-webface/`.
2. Download and extract **LFW**, **CALFW**, **CPLFW** into `data/raw/`:
   - `data/raw/lfw/lfw-deepfunneled/` + `data/raw/lfw/pairs.csv`
   - `data/raw/calfw/calfw/` + `pairs.csv`
   - `data/raw/cplfw/cplfw/` + `pairs.csv`

### Step 1 — Extract CASIA-WebFace (training data)

Decodes the `.rec` pack and saves resized JPGs for S/M/L variants:

```powershell
python scripts/extract_casia_webface.py
```

Smoke test (first 1000 images):

```powershell
python scripts/extract_casia_webface.py --max-images 1000
```

Output: `data/processed/casia-webface/{S,M,L}/{identity_id}/*.jpg`

### Step 2 — MTCNN preprocess evaluation datasets

Aligns faces in LFW / CALFW / CPLFW for validation and testing:

```powershell
python scripts/mtcnn_preprocess.py
```

Output: `data/processed/{lfw,calfw,cplfw}/{S,M,L}/`

### Step 3 — Train baseline (Contrastive / Siamese)

```powershell
python scripts/train_baseline.py --config configs/baseline_s.yaml
python scripts/train_baseline.py --config configs/baseline_m.yaml
python scripts/train_baseline.py --config configs/baseline_l.yaml
```

- Trains on random CASIA-WebFace pairs
- Validates on CALFW + CPLFW (early stopping on val EER)
- Saves to `outputs/checkpoints/baseline_{s,m,l}.pt`

### Step 4 — Train ArcFace + Hard Mining

```powershell
python scripts/train_arcface.py --config configs/arcface_s.yaml
python scripts/train_arcface.py --config configs/arcface_m.yaml
python scripts/train_arcface.py --config configs/arcface_l.yaml
```

- Trains on CASIA-WebFace identities (ArcFace + optional hard mining)
- Validates on CALFW + CPLFW
- Saves to `outputs/checkpoints/arcface_{s,m,l}.pt`

### Step 5 — Evaluate on LFW test set

```powershell
python scripts/evaluate_lfw.py
```

Runs all 6 models on LFW with horizontal-flip TTA (`lfw_test_augmentation: true` in configs).

### Step 6 — Error analysis & demo

```powershell
python scripts/error_analysis.py
python demo/verify.py
```

---

## Training Hyperparameters

| Parameter | Value |
|---|---|
| Epochs | 30 |
| Learning rate | 1e-4 |
| Optimizer | AdamW |
| Early stopping | on CALFW+CPLFW val EER |
| ArcFace | s=64, m=0.5 |
| Batch sizes | S=16, M=8, L=4 |

**Target:** EER < 5% on the LFW 6,000-pair benchmark.
