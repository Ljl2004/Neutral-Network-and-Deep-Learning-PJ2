# Neural Network and Deep Learning — Project 2

**姓名**: [陆靖磊]  |  **学号**: [22300680221]  |  **日期**: 2026年5月

**GitHub代码链接**: [[Ljl2004/Neutral-Network-and-Deep-Learning-PJ2: PJ2](https://github.com/Ljl2004/Neutral-Network-and-Deep-Learning-PJ2)]

**数据集与模型权重链接**: [[Neutral_Network_PJ2 · 模型库](https://www.modelscope.cn/models/Ljl2004/Neutral_Network_PJ2/files)]

---

## 摘要

本报告记录了《神经网络与深度学习》课程Project 2的全部实验与结果。本项目包含两部分：(1) 在CIFAR-10数据集上训练自定义神经网络以优化分类性能；(2) 研究Batch Normalization (BN) 如何帮助优化过程。

在Part 1中，我们设计了多种CNN架构（CustomCNN和CustomResNet），系统比较了不同模型大小、损失函数、激活函数和优化器的影响。最佳模型（CustomCNN-Medium + Adam优化器）在使用10,000训练样本和8个epoch的条件下，达到了**70.47%的验证准确率**。

在Part 2中，我们实现了带BN和不带BN的VGG-A模型，对比实验表明BN显著提升了分类性能（56.50% → 68.48%，提升约12个百分点），并且BN通过平滑优化景观使训练更稳定、收敛更快。

---

## Part 1: 在CIFAR-10上训练自定义网络 (60%)

### 1.1 数据集与实验设置

**CIFAR-10数据集**包含60,000张32×32彩色图像，共10个类别（飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车），每类6,000张。训练集50,000张，测试集10,000张。

**实验配置**：
- 训练子集：10,000张（加速CPU训练）
- 数据增强：RandomCrop(32, padding=4) + RandomHorizontalFlip
- 归一化：均值=(0.4914, 0.4822, 0.4465)，标准差=(0.2470, 0.2435, 0.2616)
- Batch Size：64
- Epochs：8
- 学习率调度器：CosineAnnealingLR
- 随机种子：42（所有实验一致，确保可复现）

### 1.2 网络组件

#### 必备组件（全部实现）：

| 组件 | 实现方式 |
|------|---------|
| **全连接层 (Fully-Connected)** | 分类器中使用 `nn.Linear`，如 `Linear(256→128)` 和 `Linear(128→10)` |
| **2D卷积层 (Conv2D)** | 所有特征提取阶段使用 `nn.Conv2d`，kernel_size=3，padding=1 |
| **2D池化层 (Pooling)** | `nn.MaxPool2d(kernel_size=2, stride=2)` 用于空间下采样 |
| **激活函数 (Activations)** | 支持 ReLU、LeakyReLU、GELU 多种激活函数 |

#### 可选组件（全部实现，超过最低要求）：

| 组件 | 实现方式 |
|------|---------|
| **Batch Normalization** | 每个卷积层后添加 `nn.BatchNorm2d` |
| **Dropout** | 全连接层前使用 `nn.Dropout(rate=0.3)` 防止过拟合 |
| **残差连接 (Residual Connection)** | CustomResNet中实现ResidualBlock，包含跳跃连接 |

### 1.3 网络架构设计

我们设计了四种模型变体：

| 模型 | 滤波器配置 | FC维度 | 参数量 |
|------|-----------|--------|--------|
| **CNN-Small** | (32, 64, 128) | 128 | 305,258 |
| **CNN-Medium** | (64, 128, 256) | 256 | 1,214,666 |
| **CNN-Large** | (64, 128, 256, 512) | 512 | 4,955,082 |
| **ResNet-Medium** | (64, 128, 256) + residual blocks | 256 | 2,843,466 |

### 1.4 实验一：不同架构/滤波器数量对比 (8%)

比较四种架构使用SGD+Momentum优化器、CrossEntropyLoss和ReLU激活函数的训练结果。

![架构对比](figures/p1_arch.png)

**结果分析**：

| 模型 | 最佳验证准确率 | 最终训练准确率 |
|------|:-----------:|:-----------:|
| CNN-Small (305K参数) | 57.76% | 56.01% |
| CNN-Medium (1.21M参数) | 62.64% | 61.83% |
| CNN-Large (4.96M参数) | **68.95%** | 70.39% |
| ResNet-Medium (2.84M参数) | 59.66% | 59.46% |

- CNN-Large 表现最好，说明对于此任务，更大的模型容量带来更好的性能
- ResNet-Medium 尽管参数量是CNN-Medium的2.3倍，但准确率反而更低（59.66% vs 62.64%），说明在小规模训练数据下，残差连接的优势未能充分发挥
- 从CNN-Small到CNN-Large，性能提升约11个百分点，展示了模型容量的重要性

### 1.5 实验二：不同损失函数对比 (8%)

测试四种损失函数配置：

![损失函数对比](figures/p1_loss.png)

**结果分析**：

| 损失函数 | 最佳验证准确率 |
|---------|:-----------:|
| CrossEntropy (无正则化) | **63.25%** |
| CrossEntropy + L2 (5e-4) | 63.01% |
| CrossEntropy + L2 (1e-3) | 63.02% |
| CrossEntropy + LabelSmoothing (0.1) | 62.14% |

- L2正则化在此训练规模下影响很小，四种配置性能接近
- Label Smoothing 略降低准确率约1个百分点——这是预期现象，因为标签平滑防止模型过度自信，在更大规模训练中可能提升泛化能力
- 对于8个epoch的小规模训练，标准CrossEntropy已足够

### 1.6 实验三：不同激活函数对比 (8%)

比较ReLU和LeakyReLU激活函数：

![激活函数对比](figures/p1_act.png)

**结果分析**：

| 激活函数 | 最佳验证准确率 |
|---------|:-----------:|
| **ReLU** | **63.01%** |
| LeakyReLU | 62.48% |

- ReLU略优于LeakyReLU（差距约0.5个百分点）
- 两者训练曲线几乎一致，说明对于CNN架构，ReLU仍是可靠默认选择
- LeakyReLU的负斜率设计在此任务上未带来明显收益

### 1.7 实验四：不同优化器对比 (8%)

比较SGD+Momentum、Adam和AdamW三种优化器：

![优化器对比](figures/p1_opt.png)

**结果分析**：

| 优化器 | 最佳验证准确率 |
|-------|:-----------:|
| SGD + Momentum (lr=0.01) | 63.01% |
| **Adam (lr=0.001)** | **70.47%** |
| AdamW (lr=0.001) | 69.80% |

- **这是本部分最重要的发现**：Adam和AdamW显著优于SGD+Momentum，提升了约7-8个百分点
- Adam的自适应学习率机制在有限数据和epoch条件下实现了更快的收敛和更好的最终性能
- AdamW与Adam表现接近（69.80% vs 70.47%），改进的权重衰减处理略显保守
- 这一结果验证了自适应优化器在小样本/少epoch场景下的显著优势

### 1.8 Part 1 总结

![全部结果汇总](figures/summary.png)

**主要发现**：

1. **模型容量重要**：更大的模型（更多滤波器/层数）获得更高准确率，CNN-Large达到68.95%
2. **优化器选择最关键**：Adam/AdamW比SGD提升幅度最大（+7.5个百分点），是影响性能的最重要超参数
3. **ReLU vs LeakyReLU**：ReLU略优，是可用的默认选择
4. **正则化影响小**：在小规模训练下，L2和LabelSmoothing的影响仅约1个百分点
5. **简单架构有效**：在小样本场景下，简单CNN可能优于ResNet

**最佳模型**: CustomCNN-Medium + Adam优化器 → **验证准确率 70.47%**

---

## Part 2: Batch Normalization (30%)

### 2.1 VGG-A 架构

VGG-A包含8个卷积层和3个全连接层。我们针对CIFAR-10（32×32×3输入）调整了线性层大小。

| 模型 | 参数量 |
|------|:------:|
| VGG-A (无BN) | 9,750,922 |
| VGG-A (含BN) | 9,756,426 |

BN仅增加了约5,500个参数（+0.06%），几乎不增加模型复杂度。

### 2.2 训练性能对比 (15%)

两个模型均使用SGD+Momentum (lr=0.01)、CrossEntropyLoss 和 CosineAnnealingLR 调度器训练8个epoch。

![BN训练曲线对比](figures/bn_training.png)

**结果对比**：

| 指标 | VGG-A (无BN) | VGG-A (含BN) | 提升 |
|------|:-----------:|:-----------:|:---:|
| 最佳验证准确率 | 56.50% | **68.48%** | **+11.98%** |
| 最终训练准确率 | 56.87% | **71.58%** | +14.71% |

**分析**：
- BN将验证准确率提升了约**12个百分点**（56.50% → 68.48%），改善非常显著
- 含BN的模型收敛速度明显更快：从第1个epoch起就领先无BN模型
- BN的训练准确率和验证准确率之间的差距更小，说明BN有一定正则化效果，减轻了过拟合
- 损失曲线显示含BN模型下降更快更稳定

### 2.3 损失景观分析 (15%)

为了理解BN如何帮助优化，我们参考Santurkar et al. (2018)的方法，使用不同学习率(1e-3, 2e-3, 1e-4, 5e-4)训练模型，在每个step测量损失的变化范围。max_curve和min_curve之间的区域代表优化景观的"粗糙程度"——区域越小，景观越平滑。

![损失景观对比](figures/bn_landscape.png)

![损失景观叠加对比](figures/bn_landscape_combined.png)

**分析**：

1. **更小的损失变化范围**：VGG-A+BN（蓝色区域）的max-min损失带明显窄于VGG-A无BN（红色区域）。这说明BN使损失景观更加**Lipschitz平滑**——沿梯度方向移动时损失变化更平缓可预测。

2. **更低的绝对损失值**：含BN的平均损失始终低于无BN模型，验证了BN带来更好的优化效果。

3. **一阶梯度近似更可靠**：更窄的损失带意味着在当前参数处的局部线性近似更能准确预测附近的损失变化。这是BN帮助优化的**关键机制**——它通过重参数化使优化问题变得更好条件化（better conditioned）。

4. **支持更大学习率**：平滑的景观意味着可以使用更大的学习率而不会导致训练不稳定，这也解释了为什么BN能加速训练收敛。

### 2.4 理论理解

如 Santurkar et al. (2018) 所述，BN的有效性源于其对优化景观的平滑作用，而非最初假设的"减少内部协变量偏移"（internal covariate shift）。我们的实验结果支持这一观点：

- **BN使损失景观更平滑**（更小的Lipschitz常数），使基于梯度的优化更有效
- **梯度可预测性更好**：平滑景观意味着梯度步进方向更能预测实际损失下降
- **BN的重参数化效应**使优化问题条件更好，导致更快的收敛和更好的最终性能

---

## 总结

本项目通过系统实验探索了深度学习中的两个重要主题：

**Part 1 主要结论**：
- 自定义CNN架构（含Conv2D、BN、Dropout、残差连接）可以在CIFAR-10上达到良好性能
- **优化器选择**（Adam/AdamW vs SGD）对性能影响最大（+7.5%），超过其他所有超参数
- 模型容量、激活函数和正则化策略均对最终性能有贡献，但影响程度不同

**Part 2 主要结论**：
- **Batch Normalization** 显著提升训练性能（VGG-A验证准确率绝对提升12%）
- BN通过**平滑优化景观**发挥作用，使梯度下降更有效、更可预测
- 损失景观可视化提供了BN效果的直观证据：更窄的max-min损失带 = 更平滑的景观 = 更好的优化条件

**未来工作**：可在完整CIFAR-10数据集上训练更多epoch以获得更高绝对性能；探索DenseNet/EfficientNet等更先进架构；研究LayerNorm/GroupNorm等其他归一化技术；进行更大规模的梯度可预测性定量分析。

---

## 附录：实验配置详情

| 参数 | 值 |
|------|-----|
| 数据集 | CIFAR-10 (10,000训练子集 + 10,000测试) |
| 数据增强 | RandomCrop(32, pad=4) + RandomHorizontalFlip |
| 归一化 | mean=(0.4914,0.4822,0.4465), std=(0.2470,0.2435,0.2616) |
| Batch Size | 64 |
| Epochs | 8 |
| LR调度器 | CosineAnnealingLR |
| 默认优化器 | SGD lr=0.01 momentum=0.9 weight_decay=5e-4 |
| 随机种子 | 42（所有实验一致） |
| 设备 | CPU (Intel) |

**代码目录结构**：
- `codes/VGG_BatchNorm/` — VGG-A与BN实验代码
- `codes/custom_cnn/` — 自定义CNN架构与实验代码
- `reports/figures/` — 所有实验结果图表
- `reports/models/` — 训练好的模型权重
