# Deep Learning-Based Face Verification

Face verification system using **InceptionResNetV1** backbone with three training configurations evaluated on the standard LFW 6,000-pair benchmark.

| Model | Loss | Hard Mining | Purpose |
|---|---|---|---|
| **Baseline** | CosinePairLoss (Siamese) | — | Simple verification baseline |
| **ArcFace Pure** | ArcFace only | — | Ablation: measure ArcFace alone |
| **ArcFace + HardPair** | ArcFace + HardPairContrastiveLoss | ✓ | Full model |

## Team Members — Group 8

| Name | Role |
|---|---|
| Le Viet Cuong | Project Manager, Evaluation & Pipeline |
| Hoang Quoc Huy | Data Engineer (MTCNN preprocessing) |
| Dinh Ha Hai | ML Engineer (Baseline — Siamese Network) |
| Nguyen Ngoc Linh | ML Engineer (ArcFace & Demo) |

---

## Data Protocol

| Split | Dataset | Purpose |
|---|---|---|
| **Train** | CASIA-WebFace | ~500k images, 10k identities — identity classification |
| **Val** | CALFW + CPLFW | 12,000 official pairs — early stopping on EER |
| **Test** | LFW | 6,000 official pairs — final benchmark (held-out) |

```
data/raw/lfw/                      # LFW deepfunneled + pairs.csv
data/raw/calfw/calfw/              # CALFW images + pairs.csv
data/raw/cplfw/cplfw/              # CPLFW images + pairs.csv
data/processed/
  casia-webface/I/                 # 160px MTCNN face crops for training
  lfw/I/                           # 160px MTCNN-aligned test crops
  calfw/I/                         # 160px MTCNN-aligned val crops
  cplfw/I/                         # 160px MTCNN-aligned val crops
outputs/
  checkpoints/arcface.pt           # ArcFace + Hard Pair Mining (val EER)
  checkpoints/arcface_pure.pt      # ArcFace only, no hard mining — ablation
  checkpoints/baseline.pt          # Siamese + CosinePairLoss (val EER)
  metrics/                         # ROC curves, FAR/FRR plots, comparison_table.csv
```

---

## Architecture

### Shared Backbone — InceptionResNetV1

All three models use the same backbone, pretrained on **VGGFace2** and fine-tuned on CASIA-WebFace.

```
Input image (160×160 RGB)
        │
        │  ImageNet normalization → [-1, 1] conversion
        │  (undo ImageNet norm first, then rescale to face model range)
        ▼
┌──────────────────────────────────────────────┐
│            InceptionResNetV1                 │
│  ┌────────────────────────────────────────┐  │
│  │  Stem: Conv-BN-ReLU × 3               │  │
│  ├────────────────────────────────────────┤  │
│  │  Inception-A blocks × 5               │  │
│  ├────────────────────────────────────────┤  │
│  │  Reduction-A                           │  │
│  ├────────────────────────────────────────┤  │
│  │  Inception-B blocks × 10              │  │
│  ├────────────────────────────────────────┤  │
│  │  Reduction-B                           │  │
│  ├────────────────────────────────────────┤  │
│  │  Inception-C blocks × 5               │  │
│  ├────────────────────────────────────────┤  │
│  │  AvgPool → Flatten                     │  │
│  └────────────────────────────────────────┘  │
│  Pretrained: VGGFace2  │  Unfreeze: last 30% │
└──────────────────────────────────────────────┘
        │
        │  512-dim features
        ▼
    Dropout(p=0.1)
        │
        ▼
    L2 Normalize
        │
        ▼
  512-dim unit embedding  ‖e‖ = 1
```

---

### Model A — ArcFace Pure (Ablation)

Plain ArcFace loss only — no hard mining. Used to measure the contribution of hard pair mining.

```
CASIA-WebFace identities
        │
        │  PK Batch Sampling
        │  P=32 identities × K=4 images per identity  →  batch size B = 128
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    InceptionResNetV1Backbone                    │
│              (pretrained VGGFace2, last 30% unfrozen)           │
└─────────────────────────────────────────────────────────────────┘
        │
        │  embeddings  [B × 512]  (L2-normalized)
        ▼
┌───────────────────────────────────────────────┐
│                  ArcFace Loss                 │
│                                               │
│  W ∈ ℝ^{N_cls × 512}  (class weight matrix)  │
│  L2-normalize both emb and W                 │
│                                               │
│  cos θ = emb · W^T                           │
│  θ = arccos(cos θ)                           │
│                                               │
│  Add angular margin to target class only:    │
│  cos(θ_target + m=0.5)                       │
│                                               │
│  Scale logits × s=64  →  CrossEntropy        │
│                                               │
│  Total Loss = L_arc                           │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
      ┌─────────────────────────────────────────┐
      │  SGD  momentum=0.9  weight_decay=5e-4   │
      │  ├─ backbone params:  lr = 5e-5          │
      │  └─ ArcFace head:    lr = 1e-3           │
      │  MultiStepLR: ÷2 at epoch [8, 14]        │
      │  Gradient clipping: max_norm = 5.0       │
      └─────────────────────────────────────────┘
                        │
                        ▼
      ┌─────────────────────────────────────────┐
      │  Validation  (CALFW + CPLFW, 12k pairs)  │
      │  Cosine similarity → EER                 │
      │  Early stopping patience = 15            │
      │  Save best backbone.state_dict()         │
      └─────────────────────────────────────────┘

Inference: backbone(img) → 512-d embedding → cosine similarity
```

---

### Model B — ArcFace + Hard Pair Mining (Full Model)

ArcFace with an auxiliary hard pair mining loss to additionally push the hardest same-person
pairs closer and the hardest different-person pairs further apart within each PK batch.

```
CASIA-WebFace identities
        │
        │  PK Batch Sampling
        │  P=32 identities × K=4 images per identity  →  batch size B = 128
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    InceptionResNetV1Backbone                    │
│              (pretrained VGGFace2, last 30% unfrozen)           │
└─────────────────────────────────────────────────────────────────┘
        │
        │  embeddings  [B × 512]  (L2-normalized)
        │
        ├──────────────────────────────────────────────────────────┐
        │                                                          │
        ▼                                                          ▼
┌───────────────────────────────┐              ┌──────────────────────────────────┐
│         ArcFace Loss          │              │        Hard Pair Mining          │
│                               │              │                                  │
│  cos θ = emb · W^T           │              │  Similarity matrix S = E · E^T   │
│  cos(θ_target + m=0.5)       │              │  [B × B]                         │
│  Scale × s=64                 │              │                                  │
│  CrossEntropy                 │              │  Hard Positive (per sample):     │
│                               │              │    same class, min similarity    │
│  L_arc                        │              │  Hard Negative (per sample):     │
└───────────────┬───────────────┘              │    diff class, max similarity    │
                │                              └──────────────┬───────────────────┘
                │                                             │
                │                                             ▼
                │                              ┌──────────────────────────────────┐
                │                              │    HardPairContrastiveLoss       │
                │                              │                                  │
                │                              │  pos: (τ⁺=1.0 - cos_pos)²       │
                │                              │  neg: clamp(cos_neg - τ⁻=0.5)²  │
                │                              │                                  │
                │                              │  L_hard                          │
                │                              └──────────────┬───────────────────┘
                │                                             │
                └──────────────────┬──────────────────────────┘
                                   │
                                   ▼
                   Total Loss = L_arc  +  15.0 × L_hard
                                   │
                                   ▼
             ┌─────────────────────────────────────────┐
             │  SGD  momentum=0.9  weight_decay=5e-4   │
             │  ├─ backbone params:  lr = 5e-5          │
             │  └─ ArcFace head:    lr = 1e-3           │
             │  MultiStepLR: ÷2 at epoch [8, 14]        │
             │  Gradient clipping: max_norm = 5.0       │
             └─────────────────────────────────────────┘
                                   │
                                   ▼
             ┌─────────────────────────────────────────┐
             │   Validation  (CALFW + CPLFW, 12k pairs) │
             │   Cosine similarity → EER                │
             │   Early stopping patience = 15           │
             │   Save best backbone.state_dict()        │
             └─────────────────────────────────────────┘

Inference: backbone(img) → 512-d embedding → cosine similarity
```

---

### Model C — Baseline Training

Direct verification with a Siamese Network and Cosine Pair Loss.

```
CASIA-WebFace random pairs
  (same identity / different identity, balanced 50/50)
        │
        │  ~10,000 pairs per epoch
        │  batch size = 64 pairs
        ▼
        ┌──────────────────────────────────────────────────┐
        │              SiameseNetwork                      │
        │         (shared backbone weights)                │
        │                                                  │
        │   img_A ──► InceptionResNetV1Backbone ──► emb_A  │
        │                                                  │
        │   img_B ──► InceptionResNetV1Backbone ──► emb_B  │
        │              (same weights ↑↑↑)                  │
        └───────────────────┬──────────────────────────────┘
                            │
                            │  emb_A, emb_B  [B × 512]
                            ▼
             ┌──────────────────────────────────────────┐
             │             CosinePairLoss               │
             │                                          │
             │  cos = cosine_similarity(emb_A, emb_B)   │
             │                                          │
             │  Positive pair (label=1, same person):   │
             │    loss = clamp(0.8 - cos, min=0)        │
             │    → push cos above 0.8                  │
             │                                          │
             │  Negative pair (label=0, diff person):   │
             │    loss = clamp(cos - 0.5, min=0)        │
             │    → push cos below 0.5                  │
             │                                          │
             │  Dead zone: cos ∈ [0.5, 0.8]             │
             │  (no gradient when already correct)      │
             └──────────────────┬───────────────────────┘
                                │  L_pair
                                ▼
             ┌─────────────────────────────────────────┐
             │  AdamW  lr=5e-5  weight_decay=1e-4       │
             └─────────────────────────────────────────┘
                                │
                                ▼
             ┌─────────────────────────────────────────┐
             │   Validation  (CALFW + CPLFW, 12k pairs) │
             │   Cosine similarity → EER threshold      │
             │   Early stopping patience = 8            │
             │   Save best model.state_dict()           │
             └─────────────────────────────────────────┘

Inference: backbone(img) → 512-d embedding → cosine similarity
```

---

### Inference / Verification Pipeline (all models)

```
  Live webcam frame
        │
        ▼
    MTCNN (image_size=160, margin=20)
    ├─ Face detection + alignment
    └─ 160×160 aligned crop
        │
        ▼
    ImageNet normalization
        │
        ▼
    InceptionResNetV1Backbone
        │
        ▼
    512-d L2-normalized embedding
        │
        │
  ┌─────┴──────┐
  │  emb_enroll│   (averaged from 3 enrollment captures)
  │  emb_probe │   (live verification frame)
  └─────┬──────┘
        │
        ▼
  cos_sim = emb_enroll · emb_probe
        │
        ├── cos_sim ≥ threshold  →  ✓ MATCH
        └── cos_sim < threshold  →  ✗ REJECT

  threshold = EER point on LFW 6,000-pair benchmark
              (adjustable via demo UI slider)
```

---

## Environment Setup

```bash
python -m venv bio_venv
source bio_venv/bin/activate        # Linux / WSL
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## Pipeline

### Step 1 — Extract CASIA-WebFace

Decode `.rec` file and save face crops at 160px:

```bash
python scripts/extract_casia_webface.py
```

Output: `data/processed/casia-webface/I/{identity_id}/*.jpg`

### Step 2 — MTCNN preprocess evaluation datasets

Detect and align faces in LFW / CALFW / CPLFW (saves directly at 160px):

```bash
python scripts/mtcnn_preprocess.py
```

Output: `data/processed/{lfw,calfw,cplfw}/I/`

### Step 3 — Download pre-trained weights (optional)

Initialize checkpoints from VGGFace2 pre-trained InceptionResNetV1:

```bash
python scripts/download_pretrained.py
```

### Step 4a — Train Baseline

```bash
python scripts/train_baseline.py --config configs/baseline.yaml
```

### Step 4b — Train ArcFace (pure, no hard mining)

```bash
python scripts/train_arcface.py --config configs/arcface_pure.yaml
```

Ablation: plain ArcFace loss only — used to measure the contribution of hard pair mining.

### Step 4c — Train ArcFace + Hard Pair Mining

```bash
python scripts/train_arcface.py --config configs/arcface.yaml
```

### Step 5 — Evaluate on LFW

```bash
python scripts/evaluate_lfw.py
```

Outputs EER, FAR@FRR1%, ROC curves, comparison table to `outputs/metrics/`.

### Step 6 — Error analysis

```bash
python scripts/error_analysis.py
```

### Step 7 — Run demo

```bash
python demo/app.py
# Open http://localhost:5000 in browser
```

---

## Training Hyperparameters

| Parameter | Baseline | ArcFace Pure | ArcFace + HardPair |
|---|---|---|---|
| Backbone | InceptionResNetV1 | InceptionResNetV1 | InceptionResNetV1 |
| Pre-training | VGGFace2 | VGGFace2 | VGGFace2 |
| Unfreeze ratio | 30% | 30% | 30% |
| Input size | 160 × 160 | 160 × 160 | 160 × 160 |
| Embedding dim | 512 | 512 | 512 |
| Batch | 64 pairs | PK: P=32 × K=4 = 128 | PK: P=32 × K=4 = 128 |
| Epochs | 30 | 30 | 30 |
| Optimizer | AdamW | SGD (momentum=0.9) | SGD (momentum=0.9) |
| Learning rate | 5e-5 | backbone=5e-5, head=1e-3 | backbone=5e-5, head=1e-3 |
| Weight decay | 5e-4 | 5e-4 | 5e-4 |
| LR schedule | — | MultiStepLR ÷2 at [8, 14] | MultiStepLR ÷2 at [8, 14] |
| Grad clipping | — | max_norm=5.0 | max_norm=5.0 |
| Primary loss | CosinePairLoss (τ⁺=0.8, τ⁻=0.5) | ArcFace (s=64, m=0.5) | ArcFace (s=64, m=0.5) |
| Auxiliary loss | — | — | 15.0 × HardPair (τ⁺=1.0, τ⁻=0.5) |
| Val metric | EER on CALFW+CPLFW | EER on CALFW+CPLFW | EER on CALFW+CPLFW |
| Early stopping | patience=8 | patience=15 | patience=15 |
| Freeze BN | — | Yes | Yes |
