# Deep Learning-Based Face Verification

Face verification system using **InceptionResnetV1** backbone with two training paradigms:
- **Baseline**: Siamese Network + Cosine Pair Loss
- **ArcFace**: Single-branch + Additive Angular Margin Loss

Evaluated on the standard LFW 6,000-pair benchmark.

## Team Members (Group 8)

| Name | Role |
|---|---|
| Le Viet Cuong | Project Manager, Eval & Pipeline |
| Hoang Quoc Huy | Data Engineer (MTCNN preprocessing) |
| Dinh Ha Hai | ML Engineer (Baseline Siamese Network) |
| Nguyen Ngoc Linh | ML Engineer (ArcFace & Demo) |

---

## Data Protocol

| Split | Dataset | Role |
|---|---|---|
| **Train** | CASIA-WebFace | 5,000 identities × 30 images — identity classification |
| **Val** | CALFW + CPLFW | 12,000 official eval pairs — early stopping |
| **Test** | LFW | 6,000 official eval pairs — final benchmark (held-out) |

```
archive/casia-webface/          # MXNet RecordIO (train.rec, train.idx, train.lst)
data/raw/lfw/                   # LFW deepfunneled + pairs.csv
data/raw/calfw/calfw/           # CALFW images + pairs.csv
data/raw/cplfw/cplfw/           # CPLFW images + pairs.csv
data/processed/
  casia-webface/I/              # 160px face crops for training
  lfw/I/                        # 160px MTCNN-aligned test crops
  calfw/I/                      # 160px MTCNN-aligned val crops
  cplfw/I/                      # 160px MTCNN-aligned val crops
outputs/
  checkpoints/arcface.pt        # Trained ArcFace model
  checkpoints/baseline.pt       # Trained Baseline model
  metrics/                      # ROC curves, FAR/FRR plots, comparison table
```

---

## Environment Setup

```bash
python -m venv bio_venv
source bio_venv/bin/activate        # Linux/WSL
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## Pipeline

### Step 1 — Extract CASIA-WebFace (training data)

Decodes `.rec` pack and saves 160px face crops:

```bash
python scripts/extract_casia_webface.py
```

Output: `data/processed/casia-webface/S/{identity_id}/*.jpg` (300px, before resize)

### Step 2 — MTCNN preprocess evaluation datasets

Detects and aligns faces in LFW / CALFW / CPLFW:

```bash
python scripts/mtcnn_preprocess.py
```

Output: `data/processed/{lfw,calfw,cplfw}/S/`

### Step 3 — Resize all crops to 160px

Converts all 300px crops to 160px — only needs to run once:

```bash
python scripts/resize_to_160.py
```

Output: `data/processed/{lfw,calfw,cplfw,casia-webface}/I/`  
After this step, the `S/` folders can be deleted.

### Step 4a — Train Baseline (Siamese + CosinePairLoss)

```bash
python scripts/train_baseline.py --config configs/baseline.yaml
```

- Trains on random CASIA-WebFace pairs
- Validates on CALFW + CPLFW (early stopping on val EER)
- Saves to `outputs/checkpoints/baseline.pt`

### Step 4b — Train ArcFace

```bash
python scripts/train_arcface.py --config configs/arcface.yaml
```

- Trains on CASIA-WebFace with PK batch sampling
- Uses SGD + MultiStepLR scheduler + gradient clipping
- Validates on CALFW + CPLFW
- Saves to `outputs/checkpoints/arcface.pt`

### Step 5 — Evaluate on LFW test set

```bash
python scripts/evaluate_lfw.py
```

Runs both models on LFW with horizontal-flip TTA. Outputs EER, FAR@FRR1%, ROC curves.

### Step 6 — Error analysis

```bash
python scripts/error_analysis.py
```

---

## Model Architecture

```
Input (160×160 RGB)
  ↓  ImageNet-norm → [-1,1] conversion
InceptionResnetV1 (pretrained: CASIA-WebFace)
  ↓  512-d features
Dropout(0.1)
  ↓
L2 Normalize  →  512-d unit embedding

[Baseline]                    [ArcFace]
SiameseNetwork                ArcFaceLoss head
CosinePairLoss                (s=64, m=0.5)
```

## Training Hyperparameters

| Parameter | Baseline | ArcFace |
|---|---|---|
| Backbone | InceptionResnetV1 | InceptionResnetV1 |
| Pretrained | CASIA-WebFace | CASIA-WebFace |
| Input size | 160px | 160px |
| Epochs | 30 | 32 |
| Optimizer | AdamW (lr=1e-4) | SGD (lr=1e-4 backbone, 1e-3 head) |
| LR schedule | — | MultiStepLR [20, 28] ×0.1 |
| Unfreeze ratio | 0.3 | 0.3 |
| Batch size | 64 pairs | 32 (PK: 32 ids × 4 imgs) |
| Early stopping | val EER (patience=8) | val EER (patience=10) |

**Target:** ArcFace EER < 3% on LFW 6,000-pair benchmark.
