# Solar Filament Segmentation 2026

基于 U-Net / U-Net++ 的太阳暗条（Solar Filament）图像分割项目，用于参加 **Kaggle Solar Filament Segmentation Challenge 2026**。

---

## v0.7.0 — Test-Time Augmentation

### 实验目的

在 v0.6.0 的基础上，测试 **Test-Time Augmentation（TTA）** 是否能够进一步提高模型的预测效果。

本版本不重新训练模型，而是在预测阶段对同一张图像进行不同的空间变换，再将预测结果恢复到原始方向并进行融合。

### 基础修改

模型：

* 保持 v0.5.0 的 U-Net++
* 使用已有模型权重
* 不重新训练模型

预测：

* 原图预测
* 水平翻转预测
* 垂直翻转预测
* 水平 + 垂直翻转预测
* 对多个预测结果进行平均融合

### 主要修改

原有预测流程：

```text
Model
  ↓
Probability Map
  ↓
Threshold
  ↓
Post-processing
  ↓
Submission
```

本版本：

```text
Original Image
      ↓
 ┌────┼────┬────┐
 ↓    ↓    ↓    ↓
原图  水平  垂直  水平+垂直
 ↓    ↓    ↓    ↓
 └────┼────┴────┘
      ↓
Probability Fusion
      ↓
Threshold
      ↓
Connected Components
      ↓
Minimum Area Filtering
      ↓
Submission
```

TTA 预测结果：

```python
final_prob = (prob_original + prob_h + prob_v + prob_hv) / 4
```

### 保持不变

为了保证实验具有可比性，本版本保持以下内容不变：

* U-Net++ 网络结构
* Loss
* 输入分辨率
* 数据增强
* Optimizer
* Learning Rate
* Scheduler
* Batch Size
* AMP
* Train / Validation 划分
* 模型权重
* v0.6.0 最优 Threshold
* v0.6.0 最优 Minimum Area

因此，本版本主要研究：

> **在保持模型和后处理参数不变的情况下，加入 TTA 是否能够进一步改善最终预测结果。**

---
