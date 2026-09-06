# Solar Filament Segmentation 2026

基于 U-Net / U-Net++ 的太阳暗条（Solar Filament）图像分割项目，用于参加 **Kaggle Solar Filament Segmentation Challenge 2026**。

---

## v0.8.0 — Different Resolution Model Ensemble

### 实验目的

在 v0.7.0 的 TTA 基础上，测试**不同输入分辨率的 U-Net++ 模型进行 Ensemble** 是否能够进一步提高最终预测效果。

不同输入分辨率的模型具有不同的特征提取能力：

* 低分辨率模型具有更低的计算成本，同时能够学习整体结构
* 高分辨率模型能够保留更多细节，对较小或较细的 Filament 更敏感

因此，本版本使用两个不同输入分辨率训练得到的 U-Net++ 模型进行预测融合。

### 基础修改

模型：

* 使用两个 U-Net++ 模型
* Model A：输入分辨率 512×512
* Model B：输入分辨率 1024×1024
* 两个模型使用各自训练得到的最优权重
* 不重新训练 Ensemble 模型

预测：

* 同一张测试图像分别输入两个模型
* 每个模型使用自己的输入分辨率
* 将两个模型输出的 Probability Map 恢复到原始图像尺寸
* 对两个 Probability Map 进行平均融合

### 主要修改

单模型预测：

```text
Test Image
    ↓
U-Net++
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
                    ┌─ U-Net++ @ 512 ──┐
Test Image ─────────┤                  ├─ Probability Map ─┐
                    └─ U-Net++ @ 1024 ─┘                  │
                                                          ↓
                                                   Average Fusion
                                                          ↓
                                                    Threshold
                                                          ↓
                                              Connected Components
                                                          ↓
                                               Minimum Area Filter
                                                          ↓
                                                     Submission
```

Probability Fusion：

```text
final_prob = (prob_512 + prob_1024) / 2
```

两个模型的 Probability Map 在融合前都会 resize 到原始测试图像尺寸。

### 后处理

保持 v0.6.0 的后处理流程：

```text
Probability Map
      ↓
Threshold = 0.4
      ↓
Binary Mask
      ↓
Connected Components
      ↓
Minimum Area = 100
      ↓
Final Filament Mask
```

Threshold 和 Minimum Area 沿用之前实验得到的参数。

### 保持不变

* U-Net++ 网络结构
* Loss
* 数据增强
* Optimizer
* Learning Rate
* Scheduler
* Batch Size
* AMP
* Train / Validation 划分
* v0.5.0 模型训练策略
* v0.6.0 后处理方法
* v0.7.0 后处理参数

本版本主要新增：

> **使用不同输入分辨率训练得到的两个 U-Net++ 模型进行 Probability Map Ensemble。**

通过融合 512×512 和 1024×1024 模型的预测结果，希望同时利用不同尺度下的特征信息，提高 Solar Filament 分割结果。

---
#### 注：本人未完成该实验，读者可自行实验验证