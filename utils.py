"""
工具模块 - 通用辅助函数

该模块提供：
1. 特征提取相关函数
2. 模型评估函数
3. 数据处理辅助函数
4. 显示和日志工具
"""

import torch
import torch.nn.functional as F
import numpy as np
from config import TrainConfig, DisplayConfig


# ============================================================================
# 特征提取相关
# ============================================================================

def extract_features_batch(images, cnn_model, batch_size=None, device=None):
    """
    批量提取CNN特征（避免内存溢出）
    
    Args:
        images (torch.Tensor): 图像张量 (N, 1, 28, 28)
        cnn_model (nn.Module): CNN特征提取器
        batch_size (int, optional): 批次大小
        device (str, optional): 计算设备
    
    Returns:
        torch.Tensor: 特征张量 (N, feature_dim)，在CPU上
    """
    if batch_size is None:
        batch_size = TrainConfig.FEATURE_BATCH_SIZE
    
    cnn_model.eval()  # 设置为评估模式
    all_features = []
    
    with torch.no_grad():  # 不计算梯度，节省内存
        for i in range(0, images.size(0), batch_size):
            # 取出一个批次
            batch = images[i:i+batch_size]
            
            # 提取特征
            features = cnn_model(batch)
            
            # 移到CPU上（节省GPU/MPS内存）
            all_features.append(features.cpu())
    
    # 合并所有批次的特征
    return torch.cat(all_features, dim=0)


def apply_projection(features, projection_matrix):
    """
    应用投影矩阵降维
    
    Args:
        features (torch.Tensor): 输入特征 (N, input_dim)
        projection_matrix (torch.Tensor): 投影矩阵 (input_dim, output_dim)
    
    Returns:
        torch.Tensor: 投影后的特征 (N, output_dim)
    """
    return features @ projection_matrix


def apply_dropout_mask(features, dropout_p):
    """
    应用Dropout掩码（用于闭式解训练）
    
    Args:
        features (torch.Tensor): 输入特征
        dropout_p (float): Dropout概率
    
    Returns:
        torch.Tensor: 应用Dropout后的特征
    """
    if dropout_p > 0:
        # 生成随机掩码
        mask = (torch.rand(features.shape) > dropout_p).float()
        # 应用掩码并缩放（保持期望值不变）
        features = features * mask / (1.0 - dropout_p)
    return features


# ============================================================================
# 模型评估相关
# ============================================================================

def evaluate_accuracy(model, dataloader, device):
    """
    评估模型准确率
    
    Args:
        model (nn.Module): 待评估的模型
        dataloader: 数据加载器
        device: 计算设备
    
    Returns:
        float: 准确率 (0-1之间)
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 获取预测类别
            _, predicted = outputs.max(1)
            
            # 统计正确预测数
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = correct / total
    return accuracy


def evaluate_loss(model, dataloader, criterion, device):
    """
    评估模型损失
    
    Args:
        model (nn.Module): 待评估的模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 计算设备
    
    Returns:
        float: 平均损失
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


def compute_confusion_matrix(model, dataloader, num_classes, device):
    """
    计算混淆矩阵
    
    Args:
        model (nn.Module): 模型
        dataloader: 数据加载器
        num_classes (int): 类别数
        device: 计算设备
    
    Returns:
        numpy.ndarray: 混淆矩阵 (num_classes, num_classes)
    """
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            # 更新混淆矩阵
            for true, pred in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                confusion[true, pred] += 1
    
    return confusion


# ============================================================================
# 数据处理相关
# ============================================================================

def dataloader_to_tensor(dataloader, device):
    """
    将DataLoader转换为单个张量（用于闭式解训练）
    
    注意：这会将所有数据加载到内存中，确保有足够空间
    
    Args:
        dataloader: PyTorch DataLoader
        device: 目标设备
    
    Returns:
        tuple: (images, labels) 两个张量
    """
    X_list, Y_list = [], []
    
    for images, labels in dataloader:
        X_list.append(images.to(device))
        Y_list.append(labels.to(device))
    
    X = torch.cat(X_list, dim=0)  # (N, 1, 28, 28)
    Y = torch.cat(Y_list, dim=0)  # (N,)
    
    return X, Y


def labels_to_onehot(labels, num_classes):
    """
    将标签转换为one-hot编码
    
    Args:
        labels (torch.Tensor): 标签张量 (N,)
        num_classes (int): 类别数
    
    Returns:
        torch.Tensor: one-hot张量 (N, num_classes)
    """
    return F.one_hot(labels, num_classes=num_classes).float()


# ============================================================================
# 随机投影矩阵生成
# ============================================================================

def create_random_projection(input_dim, output_dim, initialization='he'):
    """
    创建随机投影矩阵
    
    Args:
        input_dim (int): 输入维度
        output_dim (int): 输出维度
        initialization (str): 初始化方法 ('he', 'xavier', 'normal')
    
    Returns:
        torch.Tensor: 投影矩阵 (input_dim, output_dim)
    """
    if initialization == 'he':
        # He初始化（适合ReLU）
        std = np.sqrt(2.0 / input_dim)
    elif initialization == 'xavier':
        # Xavier初始化（适合tanh/sigmoid）
        std = np.sqrt(1.0 / input_dim)
    else:
        # 标准正态分布
        std = 1.0
    
    projection = torch.randn(input_dim, output_dim) * std
    return projection


# ============================================================================
# 显示和日志相关
# ============================================================================

def print_separator(title=None, length=None):
    """
    打印分隔线
    
    Args:
        title (str, optional): 标题
        length (int, optional): 分隔线长度
    """
    if length is None:
        length = DisplayConfig.SEPARATOR_LENGTH
    
    if title is None:
        print("=" * length)
    else:
        # 居中显示标题
        padding = (length - len(title) - 2) // 2
        print("=" * padding + f" {title} " + "=" * padding)


def format_accuracy(acc):
    """
    格式化准确率显示
    
    Args:
        acc (float): 准确率 (0-1)
    
    Returns:
        str: 格式化的字符串 "0.9234 (92.34%)"
    """
    return f"{acc:.4f} ({acc*100:.2f}%)"


def format_time(seconds):
    """
    格式化时间显示
    
    Args:
        seconds (float): 秒数
    
    Returns:
        str: 格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}小时"


def print_architecture(arch, title="架构配置"):
    """
    美化打印架构配置
    
    Args:
        arch (dict): 架构字典
        title (str): 标题
    """
    print(f"\n{title}:")
    print("-" * 40)
    for key, value in arch.items():
        print(f"  {key:20s}: {value}")
    print("-" * 40)


def print_training_summary(phase, train_acc, val_acc, test_acc=None):
    """
    打印训练总结
    
    Args:
        phase (str): 阶段名称
        train_acc (float): 训练集准确率
        val_acc (float): 验证集准确率
        test_acc (float, optional): 测试集准确率
    """
    print_separator(f"{phase} 训练总结")
    print(f"训练集准确率: {format_accuracy(train_acc)}")
    print(f"验证集准确率: {format_accuracy(val_acc)}")
    if test_acc is not None:
        print(f"测试集准确率: {format_accuracy(test_acc)}")
    print("=" * DisplayConfig.SEPARATOR_LENGTH)


# ============================================================================
# 模型保存和加载
# ============================================================================

def save_model(model, path, additional_info=None):
    """
    保存模型
    
    Args:
        model (nn.Module): 要保存的模型
        path (str): 保存路径
        additional_info (dict, optional): 额外信息（如配置、准确率等）
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if additional_info is not None:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, path)
    print(f"✓ 模型已保存至: {path}")


def load_model(model, path, device):
    """
    加载模型
    
    Args:
        model (nn.Module): 模型实例
        path (str): 模型路径
        device: 设备
    
    Returns:
        nn.Module: 加载权重后的模型
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"✓ 模型已从 {path} 加载")
    
    # 返回额外信息（如果有）
    extra_info = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
    return model, extra_info


# ============================================================================
# 梯度相关工具
# ============================================================================

def get_gradient_norm(model):
    """
    计算模型梯度的L2范数
    
    Args:
        model (nn.Module): 模型
    
    Returns:
        float: 梯度范数
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def clip_gradients(model, max_norm):
    """
    梯度裁剪（防止梯度爆炸）
    
    Args:
        model (nn.Module): 模型
        max_norm (float): 最大梯度范数
    
    Returns:
        float: 裁剪前的梯度范数
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


# ============================================================================
# 设置随机种子（确保可重复性）
# ============================================================================

def set_seed(seed):
    """
    设置所有随机种子
    
    Args:
        seed (int): 随机种子
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 确保确定性（可能影响性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    """
    测试工具模块
    """
    print("测试工具模块...")
    
    # 测试格式化函数
    print("\n测试格式化函数:")
    print(f"准确率: {format_accuracy(0.9234)}")
    print(f"时间: {format_time(125.5)}")
    print(f"时间: {format_time(3725)}")
    
    # 测试分隔线
    print("\n测试分隔线:")
    print_separator()
    print_separator("测试标题")
    
    # 测试架构打印
    test_arch = {
        "n_filters": 64,
        "n_conv_layers": 3,
        "use_pooling": True,
        "dropout_p": 0.1
    }
    print_architecture(test_arch)
    
    # 测试随机投影
    print("\n测试随机投影:")
    proj = create_random_projection(100, 50, 'he')
    print(f"投影矩阵形状: {proj.shape}")
    print(f"投影矩阵均值: {proj.mean():.6f}")
    print(f"投影矩阵标准差: {proj.std():.6f}")
    
    print("\n✓ 工具模块测试通过！")
