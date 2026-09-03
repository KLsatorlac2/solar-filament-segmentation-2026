# Solar Filament Segmentation 2026

A baseline and experimental pipeline for the **Solar Filament Segmentation Challenge 2026**.

The project focuses on segmenting solar filaments in H-alpha images and generating instance-level segmentation predictions in the required RLE submission format.

---

## 1. Overview

Solar filaments are elongated structures observed in H-alpha solar images. The goal of this project is to develop an image segmentation pipeline that can identify and segment filament regions from solar observations.

The current implementation uses a lightweight **U-Net** as the baseline segmentation model.

### Current pipeline

```text
H-alpha Image
      ↓
Preprocessing
      ↓
U-Net
      ↓
Pixel-wise Probability Map
      ↓
Thresholding
      ↓
Binary Segmentation Mask
      ↓
Connected Components
      ↓
Individual Filament Instances
      ↓
COCO RLE Encoding
      ↓
submission.csv
```

---

## 2. Competition

**Solar Filament Segmentation Challenge 2026**

The competition evaluates both quantitative segmentation quality and qualitative pipeline quality.

### Quantitative Comparison — 70%

The competition description specifies:

* **Panoptic Quality (PQ)** as the primary evaluation metric
* Distribution of Dice scores
* Distribution of IoU scores
* Distribution of one-to-many relations between ground-truth and predicted segmentations
* Distribution of many-to-one relations between ground-truth and predicted segmentations

### Qualitative Comparison — 30%

The evaluation also considers:

* Detailed description of the complete pipeline
* Preprocessing and final prediction procedure
* Model architecture
* Apparent morphology of predicted segmentations on H-alpha images
* Code quality, modularity, and documentation

> **Note:** PQ is the primary competition metric. The current baseline implementation includes Dice and IoU validation metrics, while an exact local PQ evaluator has not yet been implemented because the available competition description does not provide enough detail to reproduce the official calculation exactly.

---

## 3. Method

### 3.1 Baseline Model

The current baseline uses a standard U-Net architecture.

```text
Input
  │
  ▼
Encoder
  │
  ├── 32 channels
  ├── 64 channels
  ├── 128 channels
  └── 256 channels
  │
  ▼
Bottleneck
  │
  └── 512 channels
  │
  ▼
Decoder
  │
  ├── 256 channels
  ├── 128 channels
  ├── 64 channels
  └── 32 channels
  │
  ▼
1×1 Convolution
  │
  ▼
1-channel segmentation logits
```

Each encoder and decoder block uses two `3×3` convolutions with BatchNorm and ReLU.

Skip connections are used to preserve spatial information from the encoder.

---

## 4. Loss Functions

The project is designed to support multiple loss functions through configuration or command-line arguments.

Currently available:

| Name         | Loss              |
| ------------ | ----------------- |
| `bce_dice`   | BCE + Dice        |
| `focal`      | Focal Loss        |
| `focal_dice` | Focal Loss + Dice |
| `tversky`    | Tversky Loss      |

The baseline uses:

```text
BCE + Dice
```

The Loss optimization experiments are conducted by changing only the loss function while keeping the other training conditions fixed.

---

## 5. Project Structure

```text
solar-filament-segmentation-2026/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   └── config.yaml
│
├── notebooks/
│   └── solar-filament-seg.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── unet.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── losses.py
│       └── utils.py
│
└── scripts/
    ├── train.py
    └── predict.py
```

### Directory descriptions

#### `configs/`

Contains experiment configuration files.

#### `src/data/`

Contains dataset loading and preprocessing code.

`dataset.py`:

* Reads COCO-style annotations
* Converts polygon/RLE annotations to binary masks
* Resizes images and masks
* Applies training augmentation
* Creates PyTorch DataLoaders

#### `src/models/`

Contains model definitions.

`unet.py` contains the baseline U-Net implementation.

#### `src/utils/`

Contains reusable utilities.

* `losses.py` — losses and Dice/IoU metrics
* `utils.py` — random seed and general utilities

#### `scripts/`

Contains executable training and prediction scripts.

* `train.py` — model training and validation
* `predict.py` — test-set inference and submission generation

---

## 6. Dataset

The competition dataset is **not included in this repository**.

The expected Kaggle dataset structure is:

```text
MAGFiLO_1.0_Kaggle_2026/
│
├── train/
│   ├── train_images/
│   │   ├── *.jpeg
│   │   └── ...
│   │
│   └── MAGFiLO_1.0_Annotations_kaggle2026_train.json
│
└── test/
    └── test_images/
        ├── *.jpeg
        └── ...
```

The training annotations use the COCO annotation format.

---

## 7. Environment

Recommended Python version:

```text
Python 3.10+
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Required packages include:

```text
torch
torchvision
opencv-python-headless
numpy
pandas
pyyaml
tqdm
pycocotools
```

---

## 8. Configuration

The default configuration is stored in:

```text
configs/config.yaml
```

Current configuration:

```yaml
seed: 42

data:
  root: null
  image_size: 512
  val_ratio: 0.15

training:
  batch_size: 4
  num_workers: 2
  epochs: 30
  learning_rate: 0.001
  weight_decay: 0.0001
  patience: 7
  loss: bce_dice

model:
  features: [32, 64, 128, 256]

output:
  dir: outputs
```

---

## 9. Training

Run training from the project root:

```bash
python -m scripts.train \
  --data_root /path/to/MAGFiLO_1.0_Kaggle_2026
```

On Kaggle:

```bash
cd /kaggle/working/solar-filament-segmentation-2026

python -m scripts.train \
  --data_root /kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026
```

The script automatically uses CUDA when a GPU is available.

---

## 10. Loss Experiments

The training script supports selecting the loss function directly from the command line.

### BCE + Dice

```bash
python -m scripts.train \
  --data_root /path/to/MAGFiLO_1.0_Kaggle_2026 \
  --output_dir outputs_v020_bce_dice \
  --loss bce_dice
```

### Focal + Dice

```bash
python -m scripts.train \
  --data_root /path/to/MAGFiLO_1.0_Kaggle_2026 \
  --output_dir outputs_v020_focal_dice \
  --loss focal_dice
```

### Tversky

```bash
python -m scripts.train \
  --data_root /path/to/MAGFiLO_1.0_Kaggle_2026 \
  --output_dir outputs_v020_tversky \
  --loss tversky
```

Available choices:

```text
bce_dice
focal
focal_dice
tversky
```

The command-line `--loss` argument overrides the value specified in `configs/config.yaml`.

This allows different experiments to be run without modifying the configuration file.

---

## 11. Training Outputs

Each training run produces:

```text
outputs/
├── best_unet.pth
└── history.json
```

### `best_unet.pth`

Contains the model weights with the best validation Dice score during training.

### `history.json`

Contains training and validation metrics for each epoch:

```json
{
  "epoch": 1,
  "train_loss": 0.0,
  "val_loss": 0.0,
  "train_dice": 0.0,
  "val_dice": 0.0,
  "train_iou": 0.0,
  "val_iou": 0.0
}
```

---

## 12. Prediction

After training, use the best checkpoint to generate the competition submission.

Example:

```bash
python -m scripts.predict \
  --data_root /path/to/MAGFiLO_1.0_Kaggle_2026 \
  --checkpoint /path/to/best_unet.pth \
  --output /path/to/submission.csv
```

On Kaggle:

```bash
python -m scripts.predict \
  --data_root /kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026 \
  --checkpoint /kaggle/working/outputs/best_unet.pth \
  --output /kaggle/working/outputs/submission.csv
```

---

## 13. Prediction Pipeline

For each test image:

```text
Test Image
    ↓
Resize to 512×512
    ↓
Normalize
    ↓
U-Net
    ↓
Sigmoid
    ↓
Probability Map
    ↓
Threshold = 0.5
    ↓
Binary Mask
    ↓
Connected Components
    ↓
Remove Small Components
    ↓
RLE Encoding
```

The current default minimum component area is:

```text
100 pixels
```

This removes very small isolated predictions that are likely to be noise.

---

## 14. Submission Format

The final competition submission is:

```text
submission.csv
```

The CSV contains:

```text
filament_id,segmentation_rle
```

Example:

```text
image_001_1,...
image_001_2,...
image_002_1,...
```

Each connected component detected by the model is treated as an individual filament instance and encoded using COCO-style RLE.

The trained model file:

```text
best_unet.pth
```

is **not** the final Kaggle submission. It is the checkpoint used to generate `submission.csv`.

---

## 15. Current Baseline

### Version 0.1.0

The current baseline uses:

```text
Model:
    U-Net

Input:
    512 × 512

Loss:
    BCE + Dice

Optimizer:
    AdamW

Learning Rate:
    0.001

Weight Decay:
    0.0001

Scheduler:
    ReduceLROnPlateau

Early Stopping:
    Patience = 7

Data Augmentation:
    Horizontal Flip
    Vertical Flip
    Random 90° Rotations
```

Validation metrics:

```text
Dice
IoU
```

---

## 16. Design Philosophy

The project follows three principles:

### 1. Modularity

Dataset, model, loss, training, and prediction logic are separated into independent modules.

### 2. Reproducibility

Experiments use fixed random seeds and explicit configuration parameters.

### 3. Controlled Optimization

Each major version changes one primary component while keeping other training conditions as consistent as possible.

This makes it easier to understand **why** a particular modification improves or decreases performance.
