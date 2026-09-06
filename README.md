# Solar Filament Segmentation 2026

基于深度学习的太阳暗条（Solar Filament）图像分割项目，用于参加 **Kaggle Solar Filament Segmentation Challenge 2026**。

---

## v2.0.0 — SegFormer

### 实验目的

在前面的 U-Net / U-Net++ / DeepLabV3+ 模型基础上，引入 **Transformer-based Segmentation Model：SegFormer**，测试 Transformer 架构在 Solar Filament Segmentation 任务上的表现。

本版本主要研究：

> **SegFormer 是否能够利用 Transformer 的全局特征建模能力，更好地分割太阳暗条这种细长、复杂的目标结构。**

---

## 模型结构

本版本使用 **SegFormer + MiT-B0 Encoder**。

整体流程：

```text
Input H-Alpha Image
        │
        ↓
   MiT-B0 Encoder
        │
        ↓
 Multi-scale Features
        │
        ↓
 SegFormer Decoder
        │
        ↓
 Segmentation Logits
        │
        ↓
   Final Upsample
        │
        ↓
 Binary Segmentation Mask
```

SegFormer 与之前的 CNN-based segmentation models 不同，主要使用 Transformer Encoder 提取图像特征。

### MiT-B0 Encoder

MiT（Mix Transformer）负责从输入图像中提取不同尺度的特征：

```text
Input
  │
  ↓
MiT Stage 1
  │
  ↓
MiT Stage 2
  │
  ↓
MiT Stage 3
  │
  ↓
MiT Stage 4
  │
  ↓
Multi-scale Features
```

这些不同尺度的特征随后输入 SegFormer Decoder。

### SegFormer Decoder

Decoder 对不同尺度的特征进行融合，并生成最终的 segmentation logits。

最终输出：

```text
[B, 1, H, W]
```

其中：

* `B`：Batch Size
* `1`：Solar Filament segmentation channel
* `H, W`：输入图像空间尺寸

---

## 主要修改

### 1. 新增 SegFormer 模型

新增：

```text
src/models/segformer.py
```

使用 Hugging Face Transformers 提供的 SegFormer 实现，并使用：

```text
nvidia/mit-b0
```

作为 MiT-B0 backbone。

---

### 2. Training Framework 支持 SegFormer

修改：

```text
scripts/train.py
```

增加：

```text
--model segformer
```

因此可以通过：

```bash
python -m scripts.train --model segformer
```

启动 SegFormer 训练。

当前训练框架支持：

```text
UNet
UNet++
DeepLabV3+
SegFormer
```

---

### 3. 输出保持统一

SegFormer 最终输出经过 Upsample 恢复到输入图像尺寸：

```text
Input
  ↓
SegFormer
  ↓
Low-resolution Logits
  ↓
Bilinear Upsampling
  ↓
Original Resolution
```

因此可以继续使用项目现有的 Loss、Validation 和 Prediction Pipeline。

---

## Pretrained

本版本使用：

```text
nvidia/mit-b0
```

作为预训练模型。

Encoder 使用预训练参数，Segmentation Head 根据当前任务重新设置为：

```text
num_labels = 1
```

用于二分类 Solar Filament Segmentation。

---

## 保持不变

为了保证不同模型之间的实验具有可比性，本版本暂时不改变：

* Dataset
* Train / Validation Split
* Image Preprocessing
* Data Augmentation
* Loss Function
* Optimizer
* Learning Rate
* Scheduler
* Batch Size
* AMP
* Validation Metrics
* Post-processing
* TTA
* Ensemble

因此本版本主要改变：

> **Segmentation Model Architecture**

而不是同时改变多个实验变量。

---

## Training

首先测试模型：

```bash
python -m scripts.train --model segformer
```

模型权重默认保存为：

```text
best_model.pth
```

---

## Prediction

SegFormer 训练完成后，将使用对应的 SegFormer checkpoint 进行预测。

预测流程：

```text
Test Image
    ↓
Resize
    ↓
SegFormer
    ↓
Probability Map
    ↓
Threshold
    ↓
Post-processing
    ↓
Final Segmentation Mask
    ↓
Submission
```

---

## Experiment Comparison

当前项目已经包含多个 segmentation architectures：

| Version | Model           | Main Architecture    |
| ------- | --------------- | -------------------- |
| v0.x    | U-Net / U-Net++ | CNN                  |
| v1.0.0  | DeepLabV3+      | ResNet50 + ASPP      |
| v2.0.0  | SegFormer       | Transformer + MiT-B0 |
| v3.0.0  | SAM             | Foundation Model     |

v2.0.0 的核心对比对象是：

```text
U-Net++
    ↓
DeepLabV3+
    ↓
SegFormer
```

通过统一的数据集、Loss 和验证流程，对不同架构进行比较。

---

## Project Structure

```text
solar-filament-segmentation-2026/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── config.yaml
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── unet.py
│   │   ├── unet_plus_plus.py
│   │   ├── deeplabv3_plus.py
│   │   └── segformer.py
│   └── utils/
│       ├── __init__.py
│       ├── losses.py
│       └── utils.py
└── scripts/
    ├── train.py
    ├── predict.py
    ├── evaluate_postprocess.py
    ├── predict_tta.py
    └── predict_ensemble.py
```

---

## Goal

本项目的主要目标不是单纯追求单次最高分，而是通过不同 segmentation architecture 和 optimization strategy 的实验，分析它们对 Solar Filament segmentation 的影响。

当前模型路线：

```text
U-Net
  ↓
U-Net++
  ↓
DeepLabV3+
  ↓
SegFormer
  ↓
SAM
```

通过逐步替换模型架构，对比：

* Segmentation Accuracy
* Thin Structure Preservation
* Small Filament Detection
* Boundary Quality
* Generalization
* Inference Efficiency

最终选择适合 Solar Filament Segmentation 的模型方案。
