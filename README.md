# Solar Filament Segmentation 2026

基于深度学习的太阳暗条（Solar Filament）图像分割项目，用于参加 **Kaggle Solar Filament Segmentation Challenge 2026**。

项目主要针对 H-Alpha 太阳图像中的 Solar Filament 进行像素级分割。

---

## Version

### v1.0.0 — DeepLabV3+

在 v0.x 系列 U-Net / U-Net++ 实验的基础上，引入 **DeepLabV3+** 作为新的分割架构。

本版本主要研究：

> **相比 U-Net++，DeepLabV3+ 是否能够利用更大的感受野和多尺度上下文信息，提高 Solar Filament 的分割效果。**

---

## Experiment Purpose

v0.x 版本主要基于：

```text
U-Net
   ↓
U-Net++
   ↓
Loss Optimization
   ↓
Resolution Optimization
   ↓
Data Augmentation
   ↓
Post-processing
   ↓
TTA
   ↓
Multi-resolution Ensemble
```

v1.0.0 开始进入新的模型架构实验阶段：

```text
U-Net++
    ↓
DeepLabV3+
```

本版本不再继续修改 U-Net++，而是使用 DeepLabV3+ 进行独立模型对比。

---

## Model

### DeepLabV3+

当前版本采用 **ResNet50** 作为 Backbone，并结合 ASPP 和 Decoder：

```text
Input Image
     │
     ↓
 ResNet50 Backbone
     │
     ├──────────────────────────────┐
     │                              │
     ↓                              ↓
Layer1                         Layer4
(Low-level)                 (High-level)
     │                              │
     ↓                              ↓
  1×1 Conv                         ASPP
     │                              │
     │                              ↓
     │                           Upsample
     │                              │
     └──────────────┬───────────────┘
                    ↓
               Concatenate
                    ↓
                 Decoder
                    ↓
             Segmentation Logits
                    ↓
              Final Upsample
                    ↓
             Segmentation Mask
```

其中：

* **Low-level Feature**：来自 ResNet50 的 Layer1，保留较丰富的空间和边缘信息。
* **High-level Feature**：来自 ResNet50 的 Layer4，包含更强的语义和上下文信息。
* **ASPP**：对 High-level Feature 进行多尺度上下文提取。
* **Decoder**：融合 High-level 和 Low-level Feature，并恢复空间细节。

---

## Backbone

使用 **ResNet50** 作为 Encoder / Backbone。

主要提取：

* Low-level features
* High-level semantic features

当前实现中：

```text
Layer1 → Low-level features
Layer2 → Intermediate features
Layer3 → High-level features
Layer4 → High-level features
```

其中最终使用：

```text
Layer1 → Low-level Feature
Layer4 → High-level Feature
```

Low-level features 主要包含：

* 边缘
* 纹理
* 局部空间信息

High-level features 主要包含：

* Filament 语义信息
* 更大的上下文信息
* 更大的感受野

---

## ASPP

DeepLabV3+ 的核心组件之一是：

**Atrous Spatial Pyramid Pooling (ASPP)**

当前实现使用不同 dilation rate 的并行卷积：

```text
                    ┌─ 1×1 Conv
                    │
High-level Feature ─┼─ 3×3 Conv, dilation=6
                    │
                    ├─ 3×3 Conv, dilation=12
                    │
                    └─ 3×3 Conv, dilation=18
                             ↓
                        Concatenate
                             ↓
                         Projection
```

通过不同 dilation rate 获取不同尺度的上下文信息。

相比普通卷积，Atrous Convolution 可以：

> 在不大幅增加计算量的情况下扩大感受野。

这对于 Solar Filament 这种具有细长结构、尺度变化明显的目标具有一定意义。

---

## Decoder

DeepLabV3+ 与 DeepLabV3 的主要区别之一是增加了 Decoder。

当前实现的完整数据流为：

```text
                         ┌─ Layer1
                         │
Input → ResNet50 ────────┤      ↓
                         │   1×1 Conv
                         │      ↓
                         │      │
                         └─ Layer4
                                ↓
                               ASPP
                                ↓
                             Upsample
                                ↓
                         ┌──────┘
                         ↓
                    Concatenate
                         ↓
                      Decoder
                         ↓
                 Segmentation Logits
                         ↓
                   Final Upsample
                         ↓
                  Segmentation Mask
```

其中：

```text
Layer1
  ↓
Low-level Feature
  ↓
1×1 Conv
  ↓
保留空间细节
```

而：

```text
Layer4
  ↓
ASPP
  ↓
Upsample
  ↓
High-level Feature
```

最后：

```text
Low-level Feature
        +
High-level Feature
        ↓
   Concatenate
        ↓
     Decoder
        ↓
Segmentation
```

这样可以同时利用：

* High-level Feature 的语义信息
* Low-level Feature 的空间细节

从而改善目标边界和细小结构的恢复。

---

## Input / Output

输入：

```text
RGB H-Alpha Image
```

模型输入尺寸根据 `configs/config.yaml` 中的：

```yaml
data:
  image_size: ...
```

确定。

输出：

```text
1-channel segmentation logits
```

经过：

```text
Sigmoid
   ↓
Probability Map
   ↓
Threshold
   ↓
Binary Mask
```

得到最终 Solar Filament segmentation mask。

---

## Training

为了保证模型之间的对比更加公平，本版本尽量保持 v0.x 的训练流程不变。

保持：

* Train / Validation split
* Loss
* Optimizer
* Learning Rate
* Weight Decay
* Batch Size
* Data Augmentation
* AMP
* Early Stopping
* Evaluation method

主要变化：

```text
v0.x

U-Net / U-Net++
        ↓
    Segmentation


v1.0.0

DeepLabV3+
        ↓
    Segmentation
```

---

## Training Command

从项目根目录运行：

```bash
python -m scripts.train \
    --model deeplabv3plus \
    --loss bce_dice
```

如果使用 YAML 中的默认配置，也可以直接：

```bash
python -m scripts.train --model deeplabv3plus
```

训练完成后保存：

```text
outputs/
└── best_model.pth
```

---

## Prediction

使用训练好的 DeepLabV3+ 模型进行预测：

```bash
python -m scripts.predict \
    --config configs/config.yaml \
    --data_root <DATA_ROOT> \
    --checkpoint outputs/best_model.pth \
    --output outputs/submission.csv
```

预测流程：

```text
DeepLabV3+
    ↓
Segmentation Logits
    ↓
Sigmoid
    ↓
Probability Map
    ↓
Threshold
    ↓
Binary Mask
    ↓
Submission
```

---

## Comparison with v0.x

### v0.x

主要模型：

```text
U-Net
U-Net++
```

特点：

* Encoder-Decoder
* Skip Connection
* U-Net++ 使用 Nested Skip Connections
* 更强调局部空间信息和多尺度特征融合

### v1.0.0

主要模型：

```text
DeepLabV3+
```

特点：

* ResNet50 Backbone
* Atrous Convolution
* ASPP
* Low-level Feature Fusion
* Decoder
* 更大的感受野
* 多尺度上下文信息

简单对比：

| Version  | Model                     | Main Feature              |
| -------- | ------------------------- | ------------------------- |
| v0.1     | U-Net                     | Encoder-Decoder           |
| v0.2     | U-Net + Loss Optimization | Loss                      |
| v0.3     | U-Net + Resolution        | Input Resolution          |
| v0.4     | U-Net + Augmentation      | Data Augmentation         |
| v0.5     | U-Net++                   | Nested Skip Connections   |
| v0.6     | U-Net++                   | Post-processing           |
| v0.7     | U-Net++                   | TTA                       |
| v0.8     | U-Net++                   | Multi-resolution Ensemble |
| **v1.0** | **DeepLabV3+**            | **ASPP + Decoder**        |

---

## Evaluation

本项目最终评价指标主要参考 Kaggle 比赛要求。

主要包括：

### Quantitative Comparison

* Panoptic Quality (PQ)
* Dice Score Distribution
* IoU Score Distribution
* One-to-many relationship
* Many-to-one relationship

由于当前比赛对 PQ 的具体计算细节仍需要进一步确认，因此当前模型实验阶段主要使用：

```text
Dice
IoU
```

进行模型之间的初步比较。

---

## Current Pipeline

当前 v1.0.0 Pipeline：

```text
H-Alpha Image
      ↓
Preprocessing
      ↓
Resize
      ↓
Normalization
      ↓
DeepLabV3+
      ↓
ResNet50 Backbone
      │
      ├───────────────┐
      ↓               ↓
  Layer1           Layer4
      ↓               ↓
  1×1 Conv          ASPP
      │               ↓
      │            Upsample
      │               │
      └───────┬───────┘
              ↓
         Concatenate
              ↓
           Decoder
              ↓
     Segmentation Logits
              ↓
        Final Upsample
              ↓
      Segmentation Mask
```

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
│   │   └── deeplabv3_plus.py
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
