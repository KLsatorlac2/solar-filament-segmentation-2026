# Solar Filament Segmentation 2026

基于 U-Net / U-Net++ 的太阳暗条（Solar Filament）图像分割项目，用于参加 **Kaggle Solar Filament Segmentation Challenge 2026**。

---

## v0.6.0 — Post-processing Optimization

### 实验目的

在 v0.5.0 的 U-Net++ 模型基础上，测试**预测后处理策略优化**是否能够进一步提高最终分割结果。

模型输出的是每个像素属于 Solar Filament 的概率：

```text
Probability Map
      ↓
Threshold
      ↓
Binary Mask
```

不同的 Threshold 会影响预测区域的大小以及细小结构的保留情况。

因此，本版本主要研究：

* Prediction Threshold
* Minimum Component Area

对最终分割结果的影响。

### 基础修改

模型：

* 保持 v0.5.0 的 U-Net++
* 不重新训练模型
* 使用已有的最佳模型权重

预测：

* 测试不同 Threshold
* 测试不同 Minimum Area
* 使用 Connected Components 提取独立 Filament

### 主要修改

原有预测流程：

```text
Model
  ↓
Probability Map
  ↓
Threshold = 0.5
  ↓
Binary Mask
  ↓
Remove Small Objects
  ↓
Submission
```

本版本：

```text
Model
  ↓
Probability Map
  ↓
Threshold Experiment
  ↓
Binary Mask
  ↓
Connected Components
  ↓
Minimum Area Filtering
  ↓
Final Filament Mask
  ↓
Submission
```

Threshold：

```text
0.3
0.4
0.5
0.6
0.7
```

Minimum Area：

```text
50
100
200
500
```

通过验证集上的 **Dice / IoU** 比较不同参数组合，选择较优的后处理参数。

---

## 保持不变

为了保证实验具有可比性，本版本保持以下内容不变：

* v0.5.0 U-Net++ 网络结构
* v0.5.0 最优 Loss
* 输入分辨率
* 数据增强
* Optimizer
* Learning Rate
* Scheduler
* AMP
* Train / Validation 划分
* 模型权重

因此，本版本主要研究：

> **通过优化预测 Threshold 和后处理参数，是否能够进一步改善最终分割结果。**

---
