# Solar Filament Segmentation 2026

基于 U-Net / U-Net++ 的太阳暗条（Solar Filament）图像分割项目，用于参加 **Kaggle Solar Filament Segmentation Challenge 2026**。

---

## v0.5.0 — U-Net++ Architecture Experiment

### 实验目的

在 v0.4.0 的基础上，测试**网络结构优化**是否能够进一步提高太阳暗条分割性能。

v0.4.0 主要优化了数据增强策略，本版本保持数据处理和训练配置基本不变，将原有的 **U-Net** 替换为 **U-Net++**。

U-Net++ 通过更加密集的嵌套 Skip Connection，使不同尺度的特征进行更加充分的融合，从而增强模型对太阳暗条**细长结构和复杂边界**的分割能力。

### 主要修改

新增模型文件：

```text
src/models/unet_plus_plus.py
```

训练时可以通过参数选择模型：

```bash
python -m scripts.train --model unet
```

或：

```bash
python -m scripts.train --model unet_plus_plus
```
### 模型结构

* Encoder
* Bottleneck
* Nested Decoder
* Dense Skip Connections
* Segmentation Head

U-Net++ 的特征变化：

```text
输入图像
    ↓
Encoder
    ↓
空间分辨率逐渐降低
    ↓
Bottleneck
    ↓
Nested Decoder
    ↓
多层 Skip Connection 特征融合
    ↓
恢复空间分辨率
    ↓
Segmentation Head
    ↓
Filament Mask
```

### 保持不变

为了保证实验具有可比性，本版本保持以下内容不变：

* v0.4.0 使用的输入分辨率
* v0.4.0 最优 Loss
* 数据增强策略
* Optimizer
* Learning Rate
* Scheduler
* Batch Size
* Multi-GPU 训练方式
* Train / Validation 划分
* Prediction 后处理

因此，本版本主要研究：

> **将 U-Net 替换为 U-Net++ 后，网络结构是否能够进一步提高太阳暗条分割性能。**

---
