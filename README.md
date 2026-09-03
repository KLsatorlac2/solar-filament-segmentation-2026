# Solar Filament Segmentation 2026

基于 U-Net 的太阳暗条（Solar Filament）图像分割项目，用于参加 **Kaggle Solar Filament Segmentation Challenge 2026**。

---

## v0.4.0 — Data Augmentation Experiment

### 实验目的

在 v0.3.0 的基础上，测试**数据增强策略优化**是否能够提高模型的泛化能力。

Solar Filament 通常表现为细长、不规则的暗色结构，不同 H-Alpha 图像之间可能存在亮度和对比度差异。

因此，本版本在原有空间增强的基础上加入**亮度和对比度增强**，模拟不同图像条件。

### 本版本修改

config 修改:

* 分辨率调整到 1024 × 1024
* batch size 调整到 8

原有数据增强：

* 水平翻转
* 垂直翻转
* 90° / 180° / 270°随机旋转

新增：

* Random Brightness
* Random Contrast

数据处理流程：

```text
原始图像
    ↓
Resize
    ↓
随机水平/垂直翻转
    ↓
随机旋转
    ↓
随机亮度调整
    ↓
随机对比度调整
    ↓
Normalize
    ↓
模型训练
```

### 具体参数

亮度变化：

```python
delta = random.randint(-25, 25)
```

对比度变化：

```python
alpha = random.uniform(0.8, 1.2)
```

这些增强只作用于 **image**，不会修改 segmentation mask。

---

## 保持不变

为了保证实验具有可比性，本版本保持以下内容不变：

* U-Net 网络结构
* v0.3.0 使用的输入分辨率
* v0.3.0 最优 Loss
* Optimizer
* Learning Rate
* Scheduler
* Train / Validation 划分
* Prediction 后处理

因此，本版本主要研究：

> **增加亮度和对比度增强后，模型的泛化能力是否得到改善。**

---