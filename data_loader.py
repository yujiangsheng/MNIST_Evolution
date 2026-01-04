"""
数据加载模块 - 处理MNIST数据集的加载和预处理

该模块负责：
1. 下载和加载MNIST数据集
2. 应用数据增强技术提升模型泛化能力
3. 将数据集划分为训练集、验证集和测试集
4. 创建PyTorch DataLoader用于批量训练
"""

import torch
from torchvision import datasets, transforms
from config import DataConfig


def get_train_transforms():
    """
    获取训练集的数据变换（包含数据增强）
    
    数据增强技术可以有效提升模型的泛化能力，包括：
    - RandomRotation: 随机旋转，模拟手写数字的角度变化
    - RandomAffine: 随机仿射变换（平移、缩放），模拟书写位置和大小变化
    - ToTensor: 将PIL图像转换为PyTorch张量
    - Normalize: 标准化，加速训练收敛
    
    Returns:
        torchvision.transforms.Compose: 组合的数据变换
    """
    if DataConfig.USE_AUGMENTATION:
        return transforms.Compose([
            # 随机旋转：在[-10°, +10°]范围内随机旋转
            transforms.RandomRotation(DataConfig.ROTATION_DEGREES),
            
            # 随机仿射变换：
            # - translate: 在图像宽高的±10%范围内平移
            # - scale: 在90%-110%范围内缩放
            transforms.RandomAffine(
                degrees=0,  # 这里不再旋转（已经用RandomRotation了）
                translate=DataConfig.TRANSLATE,
                scale=DataConfig.SCALE
            ),
            
            # 转换为张量：将[0, 255]的PIL图像转为[0, 1]的浮点张量
            transforms.ToTensor(),
            
            # 标准化：使用MNIST数据集的均值和标准差
            # 将数据分布调整为均值0、标准差1，有利于训练
            transforms.Normalize(DataConfig.MEAN, DataConfig.STD)
        ])
    else:
        # 不使用数据增强，仅做基本的转换和标准化
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(DataConfig.MEAN, DataConfig.STD)
        ])


def get_test_transforms():
    """
    获取测试集/验证集的数据变换（不包含数据增强）
    
    测试和验证时不应该使用数据增强，只需要标准化处理
    
    Returns:
        torchvision.transforms.Compose: 组合的数据变换
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(DataConfig.MEAN, DataConfig.STD)
    ])


def load_mnist_datasets(train_size=None, val_size=None):
    """
    加载并划分MNIST数据集
    
    MNIST数据集包含：
    - 训练集：60000张28x28的手写数字灰度图
    - 测试集：10000张28x28的手写数字灰度图
    
    我们将原始训练集进一步划分为训练集和验证集，用于：
    - 训练集：用于模型训练（包含数据增强）
    - 验证集：用于超参数调优和架构选择（不增强）
    - 测试集：用于最终评估（不增强）
    
    Args:
        train_size (int, optional): 训练集大小，默认使用配置文件中的值
        val_size (int, optional): 验证集大小，默认使用配置文件中的值
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset) 三个数据集对象
    """
    # 使用配置文件中的默认值
    if train_size is None:
        train_size = DataConfig.TRAIN_SIZE
    if val_size is None:
        val_size = DataConfig.VAL_SIZE
    
    # 获取数据变换
    train_transform = get_train_transforms()
    test_transform = get_test_transforms()
    
    # 加载原始MNIST训练集（60000张）
    # 训练集使用数据增强
    train_dataset_raw = datasets.MNIST(
        root=DataConfig.DATA_ROOT,
        train=True,
        download=True,  # 如果数据不存在则自动下载
        transform=train_transform
    )
    
    # 加载MNIST测试集（10000张）
    # 测试集不使用数据增强
    test_dataset = datasets.MNIST(
        root=DataConfig.DATA_ROOT,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # 划分训练集和验证集
    # 步骤1：从60000张中取出我们需要的部分（train_size + val_size）
    total_needed = train_size + val_size
    remaining = len(train_dataset_raw) - total_needed
    
    train_val_dataset, _ = torch.utils.data.random_split(
        train_dataset_raw,
        [total_needed, remaining]
    )
    
    # 步骤2：将选出的数据进一步划分为训练集和验证集
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_val_dataset,
        [train_size, val_size]
    )
    
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset, val_dataset, test_dataset,
                       batch_size_train=None,
                       batch_size_val=None,
                       batch_size_test=None,
                       num_workers=None):
    """
    创建PyTorch DataLoader
    
    DataLoader负责：
    - 批量加载数据
    - 数据打乱（训练集）
    - 多进程加载（可选）
    
    Args:
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        test_dataset: 测试数据集
        batch_size_train (int, optional): 训练批次大小
        batch_size_val (int, optional): 验证批次大小
        batch_size_test (int, optional): 测试批次大小
        num_workers (int, optional): 数据加载线程数
    
    Returns:
        tuple: (train_loader, val_loader, test_loader) 三个数据加载器
    """
    # 使用配置文件中的默认值
    if batch_size_train is None:
        batch_size_train = DataConfig.BATCH_SIZE_TRAIN
    if batch_size_val is None:
        batch_size_val = DataConfig.BATCH_SIZE_VAL
    if batch_size_test is None:
        batch_size_test = DataConfig.BATCH_SIZE_TEST
    if num_workers is None:
        num_workers = DataConfig.NUM_WORKERS
    
    # 创建训练集DataLoader
    # shuffle=True: 每个epoch开始时打乱数据，提升训练效果
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers
    )
    
    # 创建验证集DataLoader
    # shuffle=False: 验证时不需要打乱数据
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=num_workers
    )
    
    # 创建测试集DataLoader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def get_mnist_dataloaders(train_size=None, val_size=None):
    """
    一站式获取MNIST数据加载器（组合上述所有步骤）
    
    这是一个便捷函数，封装了数据集加载和DataLoader创建的全过程
    
    Args:
        train_size (int, optional): 训练集大小
        val_size (int, optional): 验证集大小
    
    Returns:
        tuple: (train_loader, val_loader, test_loader) 以及数据集统计信息
    """
    # 加载数据集
    train_dataset, val_dataset, test_dataset = load_mnist_datasets(
        train_size=train_size,
        val_size=val_size
    )
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset
    )
    
    # 打印数据集信息
    print(f"\n{'='*60}")
    print("数据集加载完成")
    print(f"{'='*60}")
    print(f"训练集: {len(train_dataset)} 张图像 "
          f"({'使用' if DataConfig.USE_AUGMENTATION else '不使用'}数据增强)")
    print(f"验证集: {len(val_dataset)} 张图像")
    print(f"测试集: {len(test_dataset)} 张图像")
    print(f"批次大小: 训练={DataConfig.BATCH_SIZE_TRAIN}, "
          f"验证={DataConfig.BATCH_SIZE_VAL}, "
          f"测试={DataConfig.BATCH_SIZE_TEST}")
    
    return train_loader, val_loader, test_loader


# ============================================================================
# 辅助函数：数据集批量转换为张量（用于闭式解训练）
# ============================================================================

def dataloader_to_tensor(dataloader, device):
    """
    将DataLoader中的所有数据转换为单个张量
    
    用于闭式解训练时，需要一次性加载所有数据
    注意：这会将所有数据加载到内存/显存中，确保有足够的空间
    
    Args:
        dataloader: PyTorch DataLoader对象
        device: 目标设备 ('cuda', 'mps', 或 'cpu')
    
    Returns:
        tuple: (X, Y) 其中X是图像张量(N,1,28,28)，Y是标签张量(N,)
    """
    X_list, Y_list = [], []
    
    # 遍历所有批次
    for imgs, labels in dataloader:
        X_list.append(imgs.to(device))
        Y_list.append(labels.to(device))
    
    # 合并所有批次
    X = torch.cat(X_list, dim=0)  # (N, 1, 28, 28)
    Y = torch.cat(Y_list, dim=0)  # (N,)
    
    return X, Y


if __name__ == "__main__":
    """
    测试数据加载功能
    运行: python data_loader.py
    """
    print("测试数据加载模块...")
    
    # 设置随机种子
    from config import SEED
    torch.manual_seed(SEED)
    
    # 加载数据
    train_loader, val_loader, test_loader = get_mnist_dataloaders()
    
    # 测试一个批次
    print(f"\n测试一个批次:")
    images, labels = next(iter(train_loader))
    print(f"图像形状: {images.shape}")  # 应该是 (batch_size, 1, 28, 28)
    print(f"标签形状: {labels.shape}")  # 应该是 (batch_size,)
    print(f"图像数值范围: [{images.min():.3f}, {images.max():.3f}]")
    print(f"标签示例: {labels[:10].tolist()}")
    
    print("\n✓ 数据加载模块测试通过！")
