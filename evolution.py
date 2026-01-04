"""
进化算法模块 - 神经架构搜索(NAS)

该模块实现了基于进化算法的神经架构搜索：
1. 随机生成初始种群
2. 评估每个架构的性能
3. 选择优秀架构进行繁殖
4. 通过突变产生新架构
5. 重复上述过程直到找到满意的架构

核心思想：
- 架构空间搜索，而非权重训练
- 使用闭式解快速评估每个架构
- 进化策略自动优化架构组合
"""

import random
import torch
from config import EvolutionConfig, ModelConfig, TrainConfig, DEVICE
from models import RandomCNN
from utils import (dataloader_to_tensor, labels_to_onehot, 
                   extract_features_batch, create_random_projection,
                   print_separator, format_accuracy, print_architecture)
import torch.nn.functional as F


# ============================================================================
# 架构采样和突变
# ============================================================================

def sample_architecture():
    """
    随机采样一个神经网络架构
    
    架构包含以下超参数：
    - n_filters: 卷积核数量（16/32/64）
    - n_conv_layers: 卷积层数（2/3）
    - use_pooling: 是否使用池化（True/False）
    - dropout_p: Dropout率（0.0/0.1/0.2/0.3）
    - extra_projection: 是否使用额外的随机投影（True/False）
    - projection_dim: 投影维度（512/1024/2048）
    
    约束条件：
    - 不使用池化时，限制层数为2（避免特征图过大）
    
    Returns:
        dict: 架构配置字典
    """
    # 随机选择卷积核数量
    n_filters = random.choice(ModelConfig.FILTER_CHOICES)
    
    # 随机选择是否使用池化
    use_pooling = random.choice([True, False])
    
    # 根据是否池化来决定层数
    if use_pooling:
        # 使用池化时，可以有更多层
        n_conv_layers = random.choice(ModelConfig.LAYER_CHOICES)
    else:
        # 不池化时，限制为2层（避免特征图过大导致内存溢出）
        n_conv_layers = EvolutionConfig.MAX_LAYERS_WITHOUT_POOLING
    
    # 随机选择Dropout率
    dropout_p = random.choice(ModelConfig.DROPOUT_CHOICES)
    
    # 随机选择是否使用额外投影
    extra_projection = random.choice([False, True])
    
    # 如果使用额外投影，随机选择投影维度
    projection_dim = random.choice(ModelConfig.PROJECTION_DIMS) if extra_projection else 512
    
    return {
        "n_filters": n_filters,
        "n_conv_layers": n_conv_layers,
        "use_pooling": use_pooling,
        "dropout_p": dropout_p,
        "extra_projection": extra_projection,
        "projection_dim": projection_dim
    }


def mutate_architecture(arch):
    """
    对架构进行突变（随机改变一个超参数）
    
    突变策略：
    - 随机选择一个参数进行修改
    - 对于数值参数，在相邻值中选择（±1步长）
    - 对于布尔参数，直接翻转
    - 保持架构约束条件
    
    Args:
        arch (dict): 原始架构
    
    Returns:
        dict: 突变后的新架构
    """
    new_arch = arch.copy()
    
    # 随机选择要突变的参数
    param = random.choice(EvolutionConfig.MUTATION_PARAMS)
    
    if param == "n_filters":
        # 卷积核数量突变：在相邻选项中移动
        choices = ModelConfig.FILTER_CHOICES
        current_idx = choices.index(arch["n_filters"])
        # 向左或向右移动一步，或保持不变
        delta = random.choice([-1, 0, 1])
        new_idx = max(0, min(len(choices) - 1, current_idx + delta))
        new_arch["n_filters"] = choices[new_idx]
    
    elif param == "n_conv_layers":
        # 卷积层数突变
        if new_arch["use_pooling"]:
            # 使用池化时，可以在2-3之间选择
            choices = ModelConfig.LAYER_CHOICES
            current_idx = choices.index(arch["n_conv_layers"]) if arch["n_conv_layers"] in choices else 0
            delta = random.choice([-1, 0, 1])
            new_idx = max(0, min(len(choices) - 1, current_idx + delta))
            new_arch["n_conv_layers"] = choices[new_idx]
        else:
            # 不使用池化时，固定为2层
            new_arch["n_conv_layers"] = EvolutionConfig.MAX_LAYERS_WITHOUT_POOLING
    
    elif param == "use_pooling":
        # 池化开关突变
        new_arch["use_pooling"] = not arch["use_pooling"]
        # 如果改为不使用池化，需要调整层数
        if not new_arch["use_pooling"]:
            new_arch["n_conv_layers"] = EvolutionConfig.MAX_LAYERS_WITHOUT_POOLING
    
    elif param == "dropout_p":
        # Dropout率突变
        choices = ModelConfig.DROPOUT_CHOICES
        current_idx = choices.index(arch["dropout_p"])
        delta = random.choice([-1, 0, 1])
        new_idx = max(0, min(len(choices) - 1, current_idx + delta))
        new_arch["dropout_p"] = choices[new_idx]
    
    else:  # extra_projection
        # 额外投影开关突变
        new_arch["extra_projection"] = not arch["extra_projection"]
        if new_arch["extra_projection"]:
            # 如果开启投影，随机选择一个投影维度
            new_arch["projection_dim"] = random.choice(ModelConfig.PROJECTION_DIMS)
    
    return new_arch


# ============================================================================
# 闭式解训练（快速评估架构）
# ============================================================================

def train_with_closed_form(X_images, Y_labels, arch, reg=None):
    """
    使用闭式解训练分类器（岭回归）
    
    流程：
    1. 创建随机CNN提取特征
    2. 可选：应用降维（如果特征维度过高）
    3. 可选：应用额外的随机投影
    4. 可选：应用Dropout
    5. 使用岭回归闭式解求解分类器权重
    
    理论基础：
    岭回归闭式解: W = (H^T H + λI)^(-1) H^T Y
    其中 H 是特征矩阵，Y 是one-hot标签，λ 是正则化系数
    
    Args:
        X_images (torch.Tensor): 输入图像 (N, 1, 28, 28)
        Y_labels (torch.Tensor): 标签 (N,)
        arch (dict): 架构配置
        reg (float, optional): 正则化系数
    
    Returns:
        tuple: (cnn, W2, projection_matrices) 
               - cnn: 特征提取器
               - W2: 分类器权重
               - projection_matrices: 投影矩阵字典
    """
    if reg is None:
        reg = TrainConfig.RIDGE_REG
    
    device = X_images.device
    
    # 步骤1: 创建随机CNN特征提取器
    cnn = RandomCNN(arch).to(device)
    projection_matrices = {}
    
    # 步骤2: 批量提取特征（避免内存溢出，结果在CPU上）
    H = extract_features_batch(X_images, cnn, 
                               batch_size=TrainConfig.FEATURE_BATCH_SIZE)
    feature_dim = H.shape[1]
    
    # 步骤3: 特征降维（如果维度过高）
    max_dim = ModelConfig.MAX_FEATURE_DIM
    if feature_dim > max_dim:
        # 使用随机投影降维
        W_reduce = create_random_projection(feature_dim, max_dim, 'he')
        projection_matrices['W_reduce'] = W_reduce
        H = H @ W_reduce
        feature_dim = max_dim
    
    # 步骤4: 额外的随机投影层（可选）
    if arch.get("extra_projection", False):
        proj_dim = min(arch.get("projection_dim", 1024), max_dim)
        W_proj = create_random_projection(feature_dim, proj_dim, 'he')
        projection_matrices['W_proj'] = W_proj
        # 投影并应用ReLU激活
        H = F.relu(H @ W_proj)
        feature_dim = proj_dim
    
    # 步骤5: Dropout（训练时随机置零）
    dropout_p = arch.get("dropout_p", 0.0)
    if dropout_p > 0:
        mask = (torch.rand(H.shape) > dropout_p).float()
        H = H * mask / (1.0 - dropout_p)  # Dropout并缩放
    
    # 步骤6: 闭式解求解分类器权重（在CPU上计算）
    Y_cpu = Y_labels.cpu()
    Y_onehot = labels_to_onehot(Y_cpu, num_classes=10)  # 转换为one-hot
    
    # 计算 H^T H 和 H^T Y
    HtH = H.T @ H
    HtY = H.T @ Y_onehot
    
    # 创建正则化矩阵
    I = torch.eye(feature_dim, dtype=H.dtype)
    
    # 求解 W = (H^T H + λI)^(-1) H^T Y
    # 使用torch.linalg.solve求解线性方程组，比直接求逆更稳定高效
    W2 = torch.linalg.solve(HtH + reg * I, HtY)
    
    return cnn, W2, projection_matrices


def evaluate_architecture(X_images, Y_labels, arch, cnn, W2, projection_matrices):
    """
    评估架构的性能（计算准确率）
    
    使用训练好的模型在给定数据上计算准确率
    
    Args:
        X_images (torch.Tensor): 图像
        Y_labels (torch.Tensor): 标签
        arch (dict): 架构配置
        cnn: CNN特征提取器
        W2: 分类器权重
        projection_matrices: 投影矩阵字典
    
    Returns:
        float: 准确率 (0-1)
    """
    # 提取特征（批量处理，返回CPU上的张量）
    H = extract_features_batch(X_images, cnn, 
                               batch_size=TrainConfig.FEATURE_BATCH_SIZE)
    
    # 应用降维矩阵（如果有）
    if 'W_reduce' in projection_matrices:
        H = H @ projection_matrices['W_reduce']
    
    # 应用额外投影矩阵（如果有）
    if 'W_proj' in projection_matrices:
        H = F.relu(H @ projection_matrices['W_proj'])
    
    # 计算logits并预测
    logits = H @ W2
    predictions = logits.argmax(dim=1)
    
    # 计算准确率
    Y_cpu = Y_labels.cpu()
    accuracy = (predictions == Y_cpu).float().mean().item()
    
    return accuracy


# ============================================================================
# 进化搜索主流程
# ============================================================================

def evolution_search(train_loader, val_loader, 
                    pop_size=None, n_generations=None, 
                    reg=None, target_acc=None):
    """
    基于进化算法的神经架构搜索
    
    进化流程：
    1. 初始化：随机生成种群
    2. 评估：用闭式解训练每个架构并在验证集上评估
    3. 选择：选出表现最好的top-k个体作为精英
    4. 繁殖：从精英中繁殖出新个体（通过突变）
    5. 重复2-4直到达到目标精度或最大代数
    
    Args:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        pop_size (int, optional): 种群大小
        n_generations (int, optional): 最大进化代数
        reg (float, optional): 正则化系数
        target_acc (float, optional): 目标准确率（达到后提前停止）
    
    Returns:
        tuple: (best_arch, best_cnn, best_W2, best_proj_mats)
               最佳架构及其对应的模型参数
    """
    # 使用配置文件中的默认值
    if pop_size is None:
        pop_size = EvolutionConfig.POPULATION_SIZE
    if n_generations is None:
        n_generations = EvolutionConfig.MAX_GENERATIONS
    if reg is None:
        reg = TrainConfig.RIDGE_REG
    if target_acc is None:
        target_acc = TrainConfig.PHASE1_TARGET_ACC
    
    # 计算精英数量
    top_k = max(2, int(pop_size * EvolutionConfig.ELITE_RATIO))
    
    # 初始化种群
    population = [sample_architecture() for _ in range(pop_size)]
    
    # 记录全局最优
    best_arch = None
    best_acc = 0.0
    best_cnn = None
    best_W2 = None
    best_proj_mats = {}
    
    # 准备数据（转换为张量）
    print("\n正在加载数据...")
    X_train, Y_train = dataloader_to_tensor(train_loader, DEVICE)
    X_val, Y_val = dataloader_to_tensor(val_loader, DEVICE)
    
    # 进化循环
    for gen in range(n_generations):
        print(f"\n{'='*70}")
        print(f"第 {gen+1}/{n_generations} 代")
        print('='*70)
        
        scored_population = []
        
        # 评估种群中的每个个体
        for i, arch in enumerate(population):
            # 用闭式解训练
            cnn, W2, proj_mats = train_with_closed_form(X_train, Y_train, arch, reg)
            
            # 在验证集上评估
            val_acc = evaluate_architecture(X_val, Y_val, arch, cnn, W2, proj_mats)
            
            # 记录结果
            scored_population.append((val_acc, arch, cnn, W2, proj_mats))
            
            # 显示进度
            print(f"  个体 {i+1:2d}/{pop_size}: "
                  f"val_acc={format_accuracy(val_acc):15s} | arch={arch}")
        
        # 按准确率排序（降序）
        scored_population.sort(key=lambda x: x[0], reverse=True)
        
        # 本代最优个体
        gen_best_acc, gen_best_arch, gen_best_cnn, gen_best_W2, gen_best_proj = scored_population[0]
        
        print(f"\n  >> 本代最优: {format_accuracy(gen_best_acc)}")
        print_architecture(gen_best_arch, "最优架构")
        
        # 更新全局最优
        if gen_best_acc > best_acc:
            best_acc = gen_best_acc
            best_arch = gen_best_arch
            best_cnn = gen_best_cnn
            best_W2 = gen_best_W2
            best_proj_mats = gen_best_proj
            print(f"  ✓ 发现新的全局最优！准确率: {format_accuracy(best_acc)}")
        
        # 检查是否达到目标精度
        if best_acc >= target_acc:
            print(f"\n{'*'*70}")
            print(f"已达到目标精度 {format_accuracy(target_acc)}，提前结束进化！")
            print(f"{'*'*70}")
            break
        
        # 选择精英
        elites = [arch for _, arch, _, _, _ in scored_population[:top_k]]
        print(f"\n  选择前 {top_k} 个精英进行繁殖...")
        
        # 繁殖新种群
        new_population = elites.copy()  # 精英直接保留
        
        # 通过突变产生新个体
        while len(new_population) < pop_size:
            parent = random.choice(elites)  # 随机选择一个精英
            child = mutate_architecture(parent)  # 突变产生后代
            new_population.append(child)
        
        population = new_population
    
    # 进化结束
    print_separator("进化搜索完成")
    print(f"最佳验证集准确率: {format_accuracy(best_acc)}")
    print_architecture(best_arch, "最佳架构")
    print("="*70)
    
    return best_arch, best_cnn, best_W2, best_proj_mats


if __name__ == "__main__":
    """
    测试进化算法模块
    """
    print("测试进化算法模块...")
    
    # 测试架构采样
    print("\n测试架构采样:")
    for i in range(3):
        arch = sample_architecture()
        print(f"\n架构 {i+1}:")
        print_architecture(arch)
    
    # 测试架构突变
    print("\n测试架构突变:")
    original = sample_architecture()
    print("\n原始架构:")
    print_architecture(original)
    
    mutated = mutate_architecture(original)
    print("\n突变后架构:")
    print_architecture(mutated)
    
    print("\n✓ 进化算法模块测试通过！")
