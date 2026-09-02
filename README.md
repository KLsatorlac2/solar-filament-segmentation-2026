# Solar Filament Segmentation 2026

A simple U-Net baseline for the Kaggle Solar Filament Segmentation Challenge 2026.

## Project structure

```text
solar-filament-segmentation-2026/
├── README.md
├── requirements.txt
├── notebooks/
│   └── solar-filament-seg.ipynb
├── src/
│   ├── data/
│   ├── models/
│   └── utils/
├── configs/
│   └── config.yaml
└── scripts/
    ├── train.py
    └── predict.py
```

## Data

The Kaggle dataset is COCO-style and is kept outside this repository:

```text
/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/
├── train/
│   ├── train_images/
│   └── MAGFiLO_1.0_Annotations_kaggle2026_train.json
└── test/
    └── test_images/
```

## Local setup

```bash
pip install -r requirements.txt
```

Train:

```bash
python scripts/train.py --config configs/config.yaml --data_root /path/to/MAGFiLO_1.0_Kaggle_2026 --output_dir outputs
```

Predict:

```bash
python scripts/predict.py --config configs/config.yaml --data_root /path/to/MAGFiLO_1.0_Kaggle_2026 --checkpoint outputs/best_unet.pth --output outputs/submission.csv
```

## Kaggle

Clone this repository into `/kaggle/working`, keep competition data under `/kaggle/input`, then run:

```python
!git clone https://github.com/<your-username>/solar-filament-segmentation-2026.git /kaggle/working/solar-filament-segmentation-2026
%cd /kaggle/working/solar-filament-segmentation-2026
!pip install -r requirements.txt
```

Train:

```python
!python scripts/train.py \
    --config configs/config.yaml \
    --data_root /kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026 \
    --output_dir /kaggle/working/outputs
```

Predict:

```python
!python scripts/predict.py \
    --config configs/config.yaml \
    --data_root /kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026 \
    --checkpoint /kaggle/working/outputs/best_unet.pth \
    --output /kaggle/working/outputs/submission.csv
```

Competition data, checkpoints and generated outputs should not be committed to Git.
