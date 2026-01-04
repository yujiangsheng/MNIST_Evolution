"""
配置文件 - 集中管理所有超参数和配置项

这个文件包含了MNIST进化学习项目的所有配置参数，方便统一管理和调整。
主要包括：
- 设备配置（GPU/MPS/CPU）
- 随机种子设置
- 数据集参数
- 模型架构参数
- 训练参数
- 进化算法参数
"""

import torch


# ============================================================================
# 设备配置
# ============================================================================

def get_device():
    """
    自动选择最佳计算设备
    
    优先级顺序：
    1. CUDA GPU（NVIDIA显卡）
    2. MPS（Apple Silicon的GPU加速）
    3. CPU
    
    Returns:
        str: 设备名称 ('cuda', 'mps', 或 'cpu')
    """
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✓ 使用设备: CUDA (GPU: {torch.cuda.get_device_name(0)})")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        print("✓ 使用设备: MPS (Apple Silicon GPU加速)")
    else:
        device = "cpu"
        print("✓ 使用设备: CPU")
    return device


DEVICE = get_device()


# ============================================================================
# 随机种子 - 确保实验可重复性
# ============================================================================

SEED = 42  # 固定随机种子，保证每次运行结果一致


# ============================================================================
# 数据集配置
# ============================================================================

class DataConfig:
    """数据集相关配置"""
    
    # 数据路径
    DATA_ROOT = "./data"
    
    # 数据集划分
    TRAIN_SIZE = 30000  # 训练集大小（从MNIST的60000中采样）
    VAL_SIZE = 5000     # 验证集大小
    # 测试集使用MNIST标准的10000张
    
    # 数据增强参数
    USE_AUGMENTATION = True    # 是否使用数据增强
    ROTATION_DEGREES = 10      # 随机旋转角度范围
    TRANSLATE = (0.1, 0.1)     # 平移范围（相对于图像尺寸）
    SCALE = (0.9, 1.1)         # 缩放范围
    
    # 归一化参数（MNIST标准值）
    MEAN = (0.1307,)
    STD = (0.3081,)
    
    # 数据加载器参数
    BATCH_SIZE_TRAIN = 128     # 训练批次大小
    BATCH_SIZE_VAL = 256       # 验证批次大小
    BATCH_SIZE_TEST = 256      # 测试批次大小
    NUM_WORKERS = 0            # 数据加载线程数（0表示主线程加载）


# ============================================================================
# 模型架构配置
# ============================================================================

class ModelConfig:
    """模型架构相关配置"""
    
    # CNN特征提取器配置范围（用于进化搜索）
    FILTER_CHOICES = [16, 32, 64]          # 可选的卷积核数量
    LAYER_CHOICES = [2, 3]                  # 可选的卷积层数
    DROPOUT_CHOICES = [0.0, 0.1, 0.2, 0.3] # 可选的dropout率
    PROJECTION_DIMS = [512, 1024, 2048]    # 可选的投影维度
    
    # 特征维度限制（防止内存溢出）
    MAX_FEATURE_DIM = 4096  # 特征向量最大维度
    
    # 微调阶段分类器配置
    CLASSIFIER_HIDDEN_DIMS = [512, 256]    # 分类器隐藏层维度
    CLASSIFIER_DROPOUT = [0.4, 0.3]        # 分类器各层dropout率
    NUM_CLASSES = 10                        # 分类数量（MNIST有10个数字）


# ============================================================================
# 训练配置
# ============================================================================

class TrainConfig:
    """训练过程相关配置"""
    
    # 阶段1：闭式解训练
    PHASE1_TARGET_ACC = 0.93   # 阶段1目标精度（93%）
    RIDGE_REG = 1e-4           # Ridge回归正则化系数
    FEATURE_BATCH_SIZE = 256   # 特征提取批次大小（避免内存溢出）
    
    # 阶段2：反向传播微调
    PHASE2_TARGET_ACC = 0.991  # 阶段2目标精度（99.1%）
    MAX_EPOCHS = 150           # 最大训练轮数
    
    # 优化器参数
    INITIAL_LR = 0.001         # 初始学习率
    WEIGHT_DECAY = 1e-4        # 权重衰减（L2正则化）
    LABEL_SMOOTHING = 0.1      # 标签平滑（防止过拟合）
    
    # 学习率调整策略
    LR_REDUCE_FACTOR = 0.5     # 学习率衰减因子
    LR_FINE_TUNE = 0.00005     # 精细调整阶段的学习率
    
    # 早停策略
    PATIENCE = 15              # 早停耐心值（验证集精度不提升的最大epoch数）
    STAGNATION_THRESHOLD = 4   # 停滞检测阈值（触发学习率调整）
    MIN_IMPROVEMENT = 0.001    # 最小有效提升（0.1%）
    
    # 过拟合检测
    OVERFIT_THRESHOLD = 0.02   # 过拟合阈值（训练集比验证集高2%以上）


# ============================================================================
# 进化算法配置
# ============================================================================

class EvolutionConfig:
    """进化算法相关配置"""
    
    # 进化参数
    POPULATION_SIZE = 10       # 种群大小
    MAX_GENERATIONS = 15       # 最大进化代数
    ELITE_RATIO = 0.33         # 精英比例（前33%被选中繁殖）
    
    # 突变参数
    MUTATION_PARAMS = [
        "n_filters",           # 卷积核数量
        "n_conv_layers",       # 卷积层数
        "use_pooling",         # 是否使用池化
        "dropout_p",           # Dropout率
        "extra_projection"     # 是否使用额外投影
    ]
    
    # 约束条件
    # 当不使用池化时，限制层数为2（避免特征图过大导致内存溢出）
    MAX_LAYERS_WITHOUT_POOLING = 2


# ============================================================================
# 显示配置
# ============================================================================

class DisplayConfig:
    """输出显示相关配置"""
    
    # 分隔线长度
    SEPARATOR_LENGTH = 70
    
    # 是否显示详细信息
    VERBOSE = True
    
    # 进度显示格式
    EPOCH_FORMAT = "Epoch {epoch:3d}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, lr={lr:.6f}{msg}"
    INDIVIDUAL_FORMAT = "  Individual {idx}: arch={arch}, val_acc={val_acc:.4f}"


# ============================================================================
# 导出配置（供其他模块使用）
# ============================================================================

__all__ = [
    'DEVICE',
    'SEED',
    'DataConfig',
    'ModelConfig',
    'TrainConfig',
    'EvolutionConfig',
    'DisplayConfig'
]
