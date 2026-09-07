# SAM Solar Filament Segmentation

基于 **Segment Anything Model (SAM ViT-B)** 的太阳暗条（Solar Filament）分割实验。

## 1. Method

原始 SAM 对 H-alpha 太阳图像的零样本分割效果较差，因此采用 **SAM Mask Decoder Fine-tuning**：

```text
Image
  ↓
SAM Image Encoder (Frozen)
  ↓
Image Embedding
  ↓
Point Prompt
  ↓
SAM Mask Decoder (Fine-tuned)
  ↓
Filament Mask
```

训练策略：

- Image Encoder：冻结
- Prompt Encoder：冻结
- Mask Decoder：训练
- Loss：`BCE + Dice Loss`
- Input Size：`1024 × 1024`
- Validation Split：15%

## 2. SAM Checkpoint

使用官方 **SAM ViT-B** checkpoint：

```text
sam_vit_b_01ec64.pth
```

## 3. Training

```bash
python -m scripts.train_sam \
  --data_root /kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026 \
  --checkpoint /kaggle/working/sam_vit_b_01ec64.pth \
  --epochs 5 \
  --batch_size 1 \
  --image_size 1024 \
  --output outputs/best_sam.pth
```

训练完成后得到：

```text
outputs/best_sam.pth
```

该文件仅保存 fine-tuned **Mask Decoder**。

## 4. Validation

使用 Ground Truth foreground point 作为 prompt：

```text
Samples:     106
Mean Dice:   0.5992
Mean IoU:    0.4353
Median Dice: 0.6059
Median IoU:  0.4347
```

> 注：该结果使用 Ground Truth Point Prompt，并不代表完全自动推理的最终效果。

## 5.Submission

测试集使用自动生成的 candidate points，再通过 fine-tuned SAM 进行分割，并转换为 COCO RLE。

```bash
python -m scripts.create_submission_sam \
  --test_dir /kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images \
  --sam_checkpoint /kaggle/working/sam_vit_b_01ec64.pth \
  --decoder_checkpoint outputs/best_sam.pth \
  --output outputs/submission.csv
```

最终文件：

```text
outputs/submission.csv
```

Submission format：

```text
filament_id,segmentation_rle
```

## 6. Structure

```text
src/
├── data/
│   └── sam_dataset.py
└── models/
    └── sam.py

scripts/
├── train_sam.py
└── create_submission_sam.py
```

**Pipeline:**

```text
SAM ViT-B
   ↓
Freeze Image/Prompt Encoder
   ↓
Fine-tune Mask Decoder
   ↓
Automatic Point Generation
   ↓
SAM Inference
   ↓
Connected Components
   ↓
COCO RLE
   ↓
submission.csv
```