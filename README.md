# Solar Filament Segmentation 2026

基于 U-Net 的太阳暗条（Solar Filament）图像分割项目，用于参加 **Kaggle Solar Filament Segmentation Challenge 2026**。

项目采用 COCO 格式标注数据，以二值语义分割的方式预测 H-Alpha 太阳图像中的 Solar Filament 区域。

---

## v0.3.0 — Input Resolution Experiment

### 实验目的

在 v0.2.0 的基础上，测试**提高输入图像分辨率**是否能够改善 Solar Filament 的分割效果。

Solar Filament 通常具有细长、低对比度的形态。较低的输入分辨率可能导致细小结构在缩放过程中丢失，因此本版本尝试提高输入尺寸。

### 本版本修改

仅修改输入分辨率：

```yaml
data:
  image_size: 768
```

即：

```text
v0.2.0: 512 × 512
        ↓
v0.3.0: 768 × 768
```

如果显存不足，则适当降低 `batch_size`。

### 保持不变

为了保证实验具有可比性，以下内容保持不变：

* U-Net 网络结构
* v0.2.0 最优 Loss
* 数据增强
* Optimizer
* Learning Rate
* Scheduler
* Train / Validation 划分
* Prediction 后处理

因此，本版本主要观察：

> **输入分辨率从 512 提升到 768 后，模型的分割性能是否得到改善。**

---