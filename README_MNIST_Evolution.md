# MNIST 进化学习系统

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

一个创新的深度学习项目，通过**神经架构搜索(NAS)** + **闭式解训练** + **反向传播微调**的两阶段策略，在MNIST数据集上达到**99%以上**的准确率。

## ✨ 项目亮点

- 🧬 **进化算法架构搜索**：自动寻找最优网络架构，无需手动调参
- ⚡ **闭式解快速训练**：使用岭回归闭式解，无需反向传播即可快速评估架构
- 🤖 **智能优化策略**：A智能监控训练过程，动态调整学习率和正则化
- 📊 **两阶段训练**：先快速搜索架构（93%），再精细微调（99.1%+）
- 🎯 **高准确率**：最终在测试集上达到99.4%以上的准确率

## 📋 目录

- [项目概述](#项目概述)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [核心原理](#核心原理)
- [配置说明](#配置说明)
- [训练流程](#训练流程)
- [实验结果](#实验结果)
- [技术细节](#技术细节)
- [常见问题](#常见问题)

## 📖 项目概述

本项目实现了一个独特的"A指导B学习"框架：

- **A（智能优化器）**：负责架构搜索、超参数优化、训练策略调整
- **B（神经网络）**：在A的指导下不断进化和优化

### 核心创新

1. **进化 + 闭式解的结合**：用进化算法搜索架构空间，用闭式解快速评估，大大加速了架构搜索过程
2. **两阶段训练策略**：
   - 阶段1：快速找到不错的架构（目标93%）
   - 阶段2：在最优架构上精细微调（目标99.1%+）
3. **自适应优化**：A持续监控训练状态，智能调整学习率、正则化等超参数

## 📁 项目结构

```
MNIST_Evolution/
├── config.py              # 配置文件（所有超参数集中管理）
├── data_loader.py         # 数据加载模块（数据增强、预处理）
├── models.py              # 模型定义（RandomCNN、FineTuneCNN）
├── evolution.py           # 进化算法模块（架构搜索、突变）
├── trainer.py             # 训练器模块（反向传播、智能优化）
├── utils.py               # 工具函数（特征提取、评估、可视化）
├── main.py                # 主程序（整体流程协调）
├── MNIST_Evolution.py     # 原始单文件版本（已废弃）
├── README_MNIST_Evolution.md  # 项目说明文档
└── data/                  # 数据目录
    └── MNIST/             # MNIST数据集
```

### 模块说明

| 模块 | 功能 | 关键类/函数 |
|------|------|------------|
| `config.py` | 集中管理所有配置参数 | `DataConfig`, `ModelConfig`, `TrainConfig`, `EvolutionConfig` |
| `data_loader.py` | 数据加载和预处理 | `get_mnist_dataloaders()`, `dataloader_to_tensor()` |
| `models.py` | 神经网络模型定义 | `RandomCNN`, `FineTuneCNN` |
| `evolution.py` | 进化算法和架构搜索 | `evolution_search()`, `sample_architecture()`, `mutate_architecture()` |
| `trainer.py` | 训练和优化策略 | `AdaptiveTrainer`, `train_with_backprop()` |
| `utils.py` | 通用工具函数 | `extract_features_batch()`, `evaluate_accuracy()` |
| `main.py` | 主程序入口 | `main()`, `phase1_evolution_search()`, `phase2_finetune()` |

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy

### 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
pip install torch torchvision numpy
```

### 运行训练

```bash
# 完整训练流程（约5-10分钟）
python main.py
```

### 测试单个模块

```bash
# 测试数据加载
python data_loader.py

# 测试模型
python models.py

# 测试进化算法
python evolution.py

# 测试训练器
python trainer.py
```

## 🧠 核心原理

### 阶段1：进化搜索 + 闭式解

#### 架构搜索空间

搜索的超参数包括：
- 卷积核数量：16, 32, 64
- 卷积层数：2, 3
- 是否使用池化：True, False
- Dropout率：0.0, 0.1, 0.2, 0.3
- 额外投影：True, False
- 投影维度：512, 1024, 2048

#### 闭式解训练

使用岭回归闭式解训练分类器：

```
W = (H^T H + λI)^(-1) H^T Y
```

其中：
- H：特征矩阵（随机CNN提取）
- Y：one-hot标签
- λ：正则化系数
- W：分类器权重

**优势**：
- 无需反向传播，训练速度快
- 数学上有保证的全局最优解
- 适合快速评估大量架构

#### 进化策略

1. **初始化**：随机生成种群（10个架构）
2. **评估**：用闭式解训练并在验证集上评估
3. **选择**：保留表现最好的33%作为精英
4. **繁殖**：从精英中突变产生新个体
5. **迭代**：重复2-4直到达到目标或最大代数

### 阶段2：反向传播微调

#### A的智能优化策略

A持续监控训练状态并动态调整：

1. **学习率调整**：
   - 进度停滞时降低学习率（×0.5）
   - 接近目标时切换到精细调整模式（lr=0.00005）

2. **正则化调整**：
   - 检测过拟合（train_acc - val_acc > 2%）
   - 自动增加权重衰减

3. **早停机制**：
   - 验证集精度15轮未提升则停止训练
   - 避免浪费计算资源

4. **标签平滑**：
   - 使用0.1的标签平滑防止过拟合
   - 提升模型泛化能力

## ⚙️ 配置说明

所有配置集中在 `config.py` 中，可根据需要修改：

### 数据配置

```python
class DataConfig:
    TRAIN_SIZE = 30000      # 训练集大小
    VAL_SIZE = 5000         # 验证集大小
    BATCH_SIZE_TRAIN = 128  # 训练批次大小
    USE_AUGMENTATION = True  # 是否使用数据增强
```

### 进化算法配置

```python
class EvolutionConfig:
    POPULATION_SIZE = 10     # 种群大小
    MAX_GENERATIONS = 15     # 最大进化代数
    ELITE_RATIO = 0.33       # 精英比例
```

### 训练配置

```python
class TrainConfig:
    PHASE1_TARGET_ACC = 0.93   # 阶段1目标精度
    PHASE2_TARGET_ACC = 0.991  # 阶段2目标精度
    INITIAL_LR = 0.001         # 初始学习率
    MAX_EPOCHS = 150           # 最大训练轮数
```

## 📊 训练流程

### 完整流程图

```
数据加载
    ↓
阶段1：进化搜索 + 闭式解
    ├─ 初始化种群（随机架构）
    ├─ 评估个体（闭式解训练）
    ├─ 选择精英
    ├─ 繁殖突变
    └─ 达到93%目标 → 输出最优架构
    ↓
阶段2：反向传播微调
    ├─ 基于最优架构创建可训练模型
    ├─ A监控训练状态
    ├─ 动态调整学习率和正则化
    └─ 达到99.1%目标
    ↓
最终测试集评估
```

### 训练输出示例

```
============================================================
阶段1: 进化搜索 + 闭式解 (目标: 93%)
============================================================

第 1/15 代
============================================================
  个体  1/10: val_acc=0.9164 (91.64%) | arch={...}
  个体  2/10: val_acc=0.8602 (86.02%) | arch={...}
  ...
  >> 本代最优: 0.9288 (92.88%)

*** 已达到目标精度 0.9300 (93.00%)，提前结束进化！***

============================================================
阶段2: 固定架构 + 反向传播训练
============================================================
Epoch   1: train_acc=0.9294, val_acc=0.9788, lr=0.001000
Epoch   2: train_acc=0.9770, val_acc=0.9828, lr=0.001000
...
Epoch   5: train_acc=0.9885, val_acc=0.9918, lr=0.000050

*** 已达到目标精度 99.10%! ***

============================================================
最终测试集准确率: 0.9944 (99.44%)
============================================================
```

## 🎯 实验结果

### 性能指标

| 指标 | 阶段1（闭式解） | 阶段2（微调） | 最终测试 |
|------|----------------|--------------|---------|
| 准确率 | 93.7% | 99.2% | **99.4%** |
| 训练时间 | ~3分钟 | ~2分钟 | ~5分钟 |
| 方法 | 进化+闭式解 | 反向传播 | - |

### 最优架构示例

```python
{
    'n_filters': 64,           # 卷积核数量
    'n_conv_layers': 3,        # 卷积层数
    'use_pooling': True,       # 使用池化
    'dropout_p': 0.1,          # Dropout率
    'extra_projection': False,  # 不使用额外投影
    'projection_dim': 2048     # 投影维度
}
```

## 🔧 技术细节

### 随机CNN特征提取

- 使用He初始化保证特征质量
- 权重固定不训练，加速架构评估
- 理论基础：随机特征理论

### 闭式解求解

- 使用`torch.linalg.solve`求解线性方程组
- 比直接矩阵求逆更稳定高效
- 支持GPU/MPS加速

### 内存优化

- 批量特征提取（避免OOM）
- 特征维度限制（max=4096）
- CPU/GPU内存协同管理

### 数据增强

- 随机旋转（±10°）
- 随机平移（±10%）
- 随机缩放（90%-110%）

## ❓ 常见问题

### Q1：训练时间太长怎么办？

A：可以调整以下配置：
- 减少`POPULATION_SIZE`（种群大小）
- 减少`MAX_GENERATIONS`（最大进化代数）
- 减少`TRAIN_SIZE`（训练集大小）

### Q2：显存不足怎么办？

A：尝试：
- 减小`BATCH_SIZE_TRAIN`
- 设置`MAX_FEATURE_DIM`为更小的值
- 使用CPU训练（会较慢）

### Q3：准确率达不到99%？

A：可以：
- 增加训练数据量（`TRAIN_SIZE`）
- 延长训练轮数（`MAX_EPOCHS`）
- 调整学习率（`INITIAL_LR`）
- 开启数据增强（`USE_AUGMENTATION=True`）

### Q4：如何保存训练好的模型？

A：在`main.py`中添加：
```python
from utils import save_model

# 训练完成后
save_model(finetune_model, 'best_model.pth', {
    'arch': best_arch,
    'test_acc': test_acc
})
```

### Q5：如何使用自己的数据集？

A：修改`data_loader.py`中的数据加载逻辑，参考MNIST的实现方式。

## �‍💻 作者与维护者

**Jiangsheng Yu**
- GitHub: [@yujiangsheng](https://github.com/yujiangsheng)

## �📝 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- PyTorch团队提供优秀的深度学习框架
- MNIST数据集的创建者
- 所有为开源社区做出贡献的开发者

## 📧 联系方式

如有问题或建议，欢迎提Issue或PR！

---

**Happy Training! 🎉**.py - 两阶段进化训练系统

## 项目概述

这是一个创新的MNIST手写数字分类系统，采用**两阶段训练范式**：
- **阶段1**: 进化算法搜索最优架构 + 闭式解训练（无反向传播）
- **阶段2**: 固定架构 + 反向传播微调，由"智能体A"持续优化训练策略

**最终性能**: 测试集精度 **99.49%**

---

## 核心创新

### 1. 两阶段训练范式

```
阶段1 (A搜索B)          阶段2 (A指导B)
┌─────────────────┐    ┌──────────────────┐
│ 进化算法        │    │ 反向传播训练     │
│ + 闭式解        │ → │ + 动态优化       │
│ (无反向传播)    │    │ (A持续指导)      │
└─────────────────┘    └──────────────────┘
   目标: 95%              目标: 99.5%
```

### 2. 智能体A的双重角色

- **阶段1**: 通过遗传算法搜索最优CNN架构
- **阶段2**: 监控训练过程，动态调整学习率和正则化策略

---

## 技术架构

### 阶段1: 进化搜索 + 闭式解

#### 随机CNN特征提取器
```python
class RandomCNN(nn.Module):
    - 固定随机权重（He初始化）
    - 不使用反向传播
    - 参数空间: {filters, conv_layers, pooling, dropout, projection}
```

#### 闭式解分类器
使用**Ridge回归**的解析解：

$$W_2 = (H^T H + \lambda I)^{-1} H^T Y$$

其中：
- $H$: CNN提取的特征矩阵 (N × feature_dim)
- $Y$: one-hot标签矩阵 (N × 10)
- $\lambda$: 正则化系数

**优势**:
- 无需反向传播，训练速度快
- 全局最优解（凸优化问题）
- 内存高效（批量处理特征提取）

#### 进化算法
- **种群规模**: 10
- **最大代数**: 15
- **选择策略**: 精英保留 + 锦标赛选择
- **变异操作**: 架构参数随机变异
- **早停条件**: 验证集精度 ≥ 95%

### 阶段2: 反向传播 + 智能优化

#### FineTuneCNN架构
```
RandomCNN (固定) → Flatten → FC(512) → BN → ReLU → Dropout(0.3)
                                     → FC(256) → BN → ReLU → Dropout(0.3)
                                     → FC(10)
```

#### A的优化策略

| 监控指标 | 触发条件 | 优化动作 |
|---------|---------|---------|
| 训练停滞 | 连续4轮改善 < 0.1% | 降低学习率 × 0.5 |
| 精细调整阈值 | 验证精度 ≥ 98.5% | lr = 0.00005 |
| 早停 | 15轮验证精度未提升 | 停止训练 |

#### 训练技巧
- **优化器**: AdamW (weight_decay=1e-4)
- **损失函数**: CrossEntropy + Label Smoothing (0.1)
- **学习率**: 初始0.001，动态衰减
- **批大小**: 128
- **最大轮数**: 150

---

## 数据增强策略

```python
训练集增强:
- RandomRotation(10°)      # 旋转±10度
- RandomAffine(translate=0.1, scale=0.9-1.1)  # 平移+缩放
- Normalize(mean=0.1307, std=0.3081)

验证/测试集:
- 仅标准化（无增强）
```

**数据划分**:
- 训练集: 30,000 样本（带增强）
- 验证集: 5,000 样本
- 测试集: 10,000 样本

---

## 实验结果

### 最优架构（阶段1自动发现）
```python
{
    'n_filters': 64,           # 64个卷积核
    'n_conv_layers': 3,        # 3层卷积
    'use_pooling': True,       # 使用池化
    'dropout_p': 0.0,          # 无dropout（阶段1）
    'extra_projection': False, # 无额外投影
    'projection_dim': 2048     # 2048维特征
}
```

### 性能指标

| 阶段 | 方法 | 验证集精度 | 测试集精度 |
|-----|------|-----------|-----------|
| 阶段1 | 进化+闭式解 | 95.36% | - |
| 阶段2 | 反向传播+A优化 | 99.38% | **99.49%** |

### 训练过程

```
阶段1: 6代进化
- Generation 0: 92.60%
- Generation 4: 95.00% ✓
- Generation 5: 95.20% ✓✓ (达到阈值，停止)

阶段2: 35轮训练
- Epoch 1:  98.20%
- Epoch 2:  98.68% [A: 精细调整，lr=0.00005]
- Epoch 10: 99.32%
- Epoch 28: 99.38% [A: 5次学习率调整]
- Epoch 35: 停止 (验证精度15轮未提升)

A优化记录:
- 学习率调整: 0.001 → 0.00005 → 0.000025 → 0.000013 → 0.000006 → 0.000003
- 总共35轮优化指导
```

---

## 关键优化技术

### 1. 内存管理
```python
# 批量处理避免OOM
def extract_features_with_cnn(X, cnn_model, batch_size=256):
    all_features = []
    for i in range(0, X.size(0), batch_size):
        batch = X[i:i+batch_size]
        features = cnn_model(batch)
        all_features.append(features.cpu())  # 移到CPU
    return torch.cat(all_features, dim=0)
```

### 2. 特征维度控制
```python
MAX_FEATURE_DIM = 4096  # 限制最大特征维度
if feature_dim > MAX_FEATURE_DIM:
    # 使用随机投影降维
    W_reduce = torch.randn(feature_dim, MAX_FEATURE_DIM) * std
    H = H @ W_reduce
```

### 3. 早停机制
- 阶段1: 达到95%自动停止进化
- 阶段2: 验证精度15轮未提升停止训练

---

## 运行环境

- **设备**: MPS (Apple Silicon)
- **Python**: 3.9+
- **依赖**: PyTorch, torchvision, numpy

### 运行方法
```bash
python3 MNIST_Evolution.py
```

### 输出示例
```
Using device: MPS (Apple Silicon)
Train set: 30000 samples (使用数据增强)
Val set: 5000 samples
Test set: 10000 samples

======================================================================
阶段1: 进化搜索 + 闭式解 (目标: 95%)
======================================================================
...
阶段1完成，验证集精度: 0.9536 (95.36%)

======================================================================
阶段2: 固定架构B，使用反向传播训练 (目标: 99.5%)
======================================================================
...
最终测试集精度: 0.9949 (99.49%)
```

---

## 代码结构

```
MNIST_Evolution.py (629行)
│
├── load_mnist()              # 数据加载+增强
│
├── 阶段1: 闭式解训练
│   ├── RandomCNN             # 随机CNN特征提取器
│   ├── extract_features_with_cnn()  # 批量特征提取
│   ├── train_B_with_closed_form()   # Ridge回归闭式解
│   ├── evaluate_B()          # 模型评估
│   └── evolution_search()    # 遗传算法搜索
│
├── 阶段2: 反向传播微调
│   ├── FineTuneCNN           # 可训练分类器
│   └── finetune_with_backprop()  # A指导的训练循环
│
└── main()                    # 两阶段训练流程
```

---

## 理论优势

### 1. 计算效率
- 阶段1无需反向传播，快速评估数千个架构
- 闭式解一步求出全局最优

### 2. 泛化能力
- 随机特征提供多样性
- 两阶段训练避免过早收敛
- 数据增强提升鲁棒性

### 3. 可解释性
- 进化过程可视化架构演变
- A的优化决策有明确逻辑
- 闭式解具有理论保证

---

## 扩展方向

1. **更大数据集**: 扩展到CIFAR-10/100
2. **迁移学习**: 预训练特征 + 闭式解分类
3. **多任务学习**: 同时优化多个目标
4. **神经架构搜索**: 更复杂的搜索空间
5. **分布式训练**: 并行评估多个候选架构

---

## 参考文献

- **Random Features**: Rahimi & Recht (2007) - "Random Features for Large-Scale Kernel Machines"
- **Extreme Learning Machine**: Huang et al. (2006)
- **Neural Architecture Search**: Zoph & Le (2017)
- **Label Smoothing**: Szegedy et al. (2016)

---

## 总结

这个项目展示了如何将**进化算法**、**闭式解优化**和**反向传播**有机结合，创建一个高效且高性能的训练系统。通过智能体A的双重角色（搜索+指导），实现了从95%到99.49%的精度提升，验证了两阶段训练范式的有效性。

**核心思想**: 先快速探索（阶段1），再精细优化（阶段2），充分发挥各方法优势。
