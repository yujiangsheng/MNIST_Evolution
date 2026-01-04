"""
MNIST进化学习 - 主程序
神经架构搜索 + 闭式解训练 + 反向传播微调

项目目标：
通过两阶段训练在MNIST数据集上达到99%以上的准确率
- 阶段1: 进化搜索最优架构 + 闭式解训练（目标: 93%）
- 阶段2: 固定架构 + 反向传播微调（目标: 99.1%）

设计理念：
模拟"A指导B学习"的过程，其中：
- A: 智能优化策略（架构搜索、学习率调整、正则化等）
- B: 神经网络模型（不断进化和优化）
"""

import torch
import random
import numpy as np

# 导入配置
from config import SEED, DEVICE, TrainConfig, DisplayConfig

# 导入数据加载
from data_loader import get_mnist_dataloaders

# 导入模型
from models import FineTuneCNN

# 导入进化搜索
from evolution import evolution_search, evaluate_architecture

# 导入训练器
from trainer import train_with_backprop

# 导入工具函数
from utils import (dataloader_to_tensor, print_separator, 
                   format_accuracy, print_architecture,
                   evaluate_accuracy, print_training_summary)


def set_all_seeds(seed=SEED):
    """
    设置所有随机种子，确保实验可重复
    
    Args:
        seed (int): 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 注意：这会影响性能，但能保证完全的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def phase1_evolution_search(train_loader, val_loader):
    """
    阶段1: 使用进化算法搜索最优架构，并用闭式解训练
    
    这个阶段的目标是快速找到一个不错的神经网络架构（目标93%准确率），
    而不需要进行耗时的反向传播训练。
    
    关键特点：
    - 架构空间搜索（进化算法）
    - 快速评估（闭式解，无需反向传播）
    - 达到目标后自动停止
    
    Args:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
    
    Returns:
        tuple: (best_arch, best_cnn, best_W2, best_proj_mats, phase1_val_acc)
    """
    print("\n" + "="*DisplayConfig.SEPARATOR_LENGTH)
    print("阶段1: 进化搜索 + 闭式解训练")
    print("="*DisplayConfig.SEPARATOR_LENGTH)
    print(f"目标: 达到 {format_accuracy(TrainConfig.PHASE1_TARGET_ACC)} 准确率")
    print(f"策略: A通过进化算法搜索最优架构，使用闭式解快速评估")
    print("="*DisplayConfig.SEPARATOR_LENGTH)
    
    # 运行进化搜索
    best_arch, best_cnn, best_W2, best_proj_mats = evolution_search(
        train_loader, val_loader
    )
    
    # 评估阶段1的最终结果
    print("\n正在评估阶段1的最终性能...")
    X_val, Y_val = dataloader_to_tensor(val_loader, DEVICE)
    phase1_val_acc = evaluate_architecture(
        X_val, Y_val, best_arch, best_cnn, best_W2, best_proj_mats
    )
    
    print_separator("阶段1完成")
    print(f"验证集准确率: {format_accuracy(phase1_val_acc)}")
    print_architecture(best_arch, "最优架构")
    print("="*DisplayConfig.SEPARATOR_LENGTH)
    
    return best_arch, best_cnn, best_W2, best_proj_mats, phase1_val_acc


def phase2_finetune(best_arch, best_cnn, train_loader, val_loader):
    """
    阶段2: 固定架构，使用反向传播微调至高精度
    
    在阶段1找到的最优架构基础上，通过反向传播训练达到更高的精度。
    A会持续监控训练过程，动态调整学习率、正则化等超参数。
    
    关键特点：
    - 端到端的反向传播训练
    - A的智能优化指导（学习率调整、早停等）
    - 目标99.1%以上准确率
    
    Args:
        best_arch: 阶段1找到的最优架构
        best_cnn: 阶段1的CNN特征提取器
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
    
    Returns:
        tuple: (finetune_model, phase2_val_acc)
    """
    print("\n" + "="*DisplayConfig.SEPARATOR_LENGTH)
    print("阶段2: 固定架构 + 反向传播微调")
    print("="*DisplayConfig.SEPARATOR_LENGTH)
    print(f"目标: 达到 {format_accuracy(TrainConfig.PHASE2_TARGET_ACC)} 准确率")
    print(f"策略: A智能监控训练过程，动态优化学习策略")
    print_architecture(best_arch, "固定架构")
    
    # 创建可训练的模型（基于阶段1的特征提取器）
    finetune_model = FineTuneCNN(best_cnn, num_classes=10).to(DEVICE)
    
    # 使用A的智能训练器进行微调
    finetune_model, phase2_val_acc = train_with_backprop(
        finetune_model, train_loader, val_loader,
        target_acc=TrainConfig.PHASE2_TARGET_ACC,
        max_epochs=TrainConfig.MAX_EPOCHS
    )
    
    return finetune_model, phase2_val_acc


def final_evaluation(model, test_loader):
    """
    最终评估：在测试集上评估模型性能
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
    
    Returns:
        float: 测试集准确率
    """
    print_separator("最终测试集评估")
    test_acc = evaluate_accuracy(model, test_loader, DEVICE)
    print(f"测试集准确率: {format_accuracy(test_acc)}")
    print("="*DisplayConfig.SEPARATOR_LENGTH)
    
    return test_acc


def main():
    """
    主函数：协调整个训练流程
    """
    print("\n" + "="*DisplayConfig.SEPARATOR_LENGTH)
    print("MNIST 进化学习 - 神经架构搜索 + 智能优化")
    print("="*DisplayConfig.SEPARATOR_LENGTH)
    print("项目目标: 通过两阶段训练达到99%以上准确率")
    print("阶段1: 进化搜索最优架构（闭式解训练）")
    print("阶段2: 反向传播微调（A智能指导B学习）")
    print("="*DisplayConfig.SEPARATOR_LENGTH)
    
    # ========================================================================
    # 1. 初始化：设置随机种子
    # ========================================================================
    print(f"\n初始化随机种子: {SEED}")
    set_all_seeds(SEED)
    
    # ========================================================================
    # 2. 数据加载
    # ========================================================================
    train_loader, val_loader, test_loader = get_mnist_dataloaders()
    
    # ========================================================================
    # 3. 阶段1: 进化搜索 + 闭式解训练
    # ========================================================================
    best_arch, best_cnn, best_W2, best_proj_mats, phase1_val_acc = \
        phase1_evolution_search(train_loader, val_loader)
    
    # 检查阶段1是否成功
    if phase1_val_acc < 0.90:
        print("\n⚠ 警告: 阶段1精度未达到90%")
        print(f"当前精度: {format_accuracy(phase1_val_acc)}")
        print("建议检查配置或增加进化代数")
        return
    
    # ========================================================================
    # 4. 阶段2: 反向传播微调
    # ========================================================================
    finetune_model, phase2_val_acc = phase2_finetune(
        best_arch, best_cnn, train_loader, val_loader
    )
    
    # ========================================================================
    # 5. 最终评估
    # ========================================================================
    test_acc = final_evaluation(finetune_model, test_loader)
    
    # ========================================================================
    # 6. 训练总结
    # ========================================================================
    print("\n" + "="*DisplayConfig.SEPARATOR_LENGTH)
    print("训练完成！完整总结")
    print("="*DisplayConfig.SEPARATOR_LENGTH)
    print(f"\n阶段1 (进化搜索 + 闭式解):")
    print(f"  验证集准确率: {format_accuracy(phase1_val_acc)}")
    print_architecture(best_arch, "  最优架构")
    
    print(f"\n阶段2 (反向传播微调):")
    print(f"  验证集准确率: {format_accuracy(phase2_val_acc)}")
    print(f"  提升幅度: +{(phase2_val_acc - phase1_val_acc)*100:.2f}%")
    
    print(f"\n最终测试结果:")
    print(f"  测试集准确率: {format_accuracy(test_acc)}")
    
    # 判断是否达到目标
    if test_acc >= 0.99:
        print(f"\n✓ 成功达到目标！测试集准确率超过99%")
    else:
        print(f"\n未完全达到99%目标，但已取得 {format_accuracy(test_acc)} 的成绩")
    
    print("\n" + "="*DisplayConfig.SEPARATOR_LENGTH)
    print("感谢使用 MNIST进化学习系统！")
    print("="*DisplayConfig.SEPARATOR_LENGTH + "\n")


if __name__ == "__main__":
    """
    程序入口
    
    直接运行此文件即可开始完整的训练流程：
    python main.py
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
    except Exception as e:
        print(f"\n\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
