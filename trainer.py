"""
训练器模块 - 处理模型训练和微调

该模块包含：
1. 反向传播训练器（用于阶段2的微调）
2. 训练状态监控
3. 自适应优化策略（A对B的智能指导）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from config import TrainConfig, DEVICE, DisplayConfig
from utils import evaluate_accuracy, print_separator, format_accuracy


class AdaptiveTrainer:
    """
    自适应训练器 - 实现A对B的智能优化指导
    
    功能：
    1. 训练模型直到达到目标精度
    2. 动态调整学习率
    3. 检测过拟合/欠拟合并调整策略
    4. 早停机制
    
    A的优化策略：
    - 进度停滞时降低学习率
    - 过拟合时增加正则化
    - 接近目标时切换到精细调整模式
    - 持续监控训练状态并给出反馈
    """
    
    def __init__(self, model, train_loader, val_loader, 
                 target_acc=None, max_epochs=None):
        """
        初始化训练器
        
        Args:
            model: 要训练的模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            target_acc (float, optional): 目标准确率
            max_epochs (int, optional): 最大训练轮数
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.target_acc = target_acc or TrainConfig.PHASE2_TARGET_ACC
        self.max_epochs = max_epochs or TrainConfig.MAX_EPOCHS
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.stagnation_counter = 0
        self.val_history = []
        
        # 优化器和损失函数
        self.current_lr = TrainConfig.INITIAL_LR
        self.current_weight_decay = TrainConfig.WEIGHT_DECAY
        self._init_optimizer()
        self._init_criterion()
    
    def _init_optimizer(self):
        """初始化优化器"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.current_lr,
            weight_decay=self.current_weight_decay
        )
    
    def _init_criterion(self):
        """初始化损失函数"""
        # 使用标签平滑防止过拟合
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=TrainConfig.LABEL_SMOOTHING
        )
    
    def train_epoch(self):
        """
        训练一个epoch
        
        Returns:
            tuple: (train_loss, train_acc) 训练损失和准确率
        """
        self.model.train()  # 设置为训练模式
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in self.train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """
        在验证集上评估
        
        Returns:
            float: 验证集准确率
        """
        val_acc = evaluate_accuracy(self.model, self.val_loader, DEVICE)
        return val_acc
    
    def update_learning_rate(self, new_lr):
        """
        更新学习率
        
        Args:
            new_lr (float): 新的学习率
        """
        old_lr = self.current_lr
        self.current_lr = new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return old_lr
    
    def update_weight_decay(self, new_wd):
        """
        更新权重衰减（需要重新创建优化器）
        
        Args:
            new_wd (float): 新的权重衰减值
        """
        old_wd = self.current_weight_decay
        self.current_weight_decay = new_wd
        self._init_optimizer()
        return old_wd
    
    def check_and_optimize(self, train_acc, val_acc):
        """
        A的核心优化逻辑：检查训练状态并调整策略
        
        Args:
            train_acc (float): 训练集准确率
            val_acc (float): 验证集准确率
        
        Returns:
            str: 优化建议消息
        """
        guidance_msg = ""
        
        # 1. 检查是否有提升
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.patience_counter = 0
            self.stagnation_counter = 0
        else:
            self.patience_counter += 1
            self.stagnation_counter += 1
        
        # 2. 检测停滞并降低学习率
        if self.stagnation_counter >= TrainConfig.STAGNATION_THRESHOLD:
            # 检查最近的进步幅度
            if len(self.val_history) >= 5:
                recent_improvement = self.val_history[-1] - self.val_history[-5]
                
                if recent_improvement < TrainConfig.MIN_IMPROVEMENT:
                    # 进步缓慢，降低学习率
                    old_lr = self.update_learning_rate(
                        self.current_lr * TrainConfig.LR_REDUCE_FACTOR
                    )
                    guidance_msg = (f" [A优化: 降低学习率 "
                                  f"{old_lr:.6f} → {self.current_lr:.6f}]")
                    self.stagnation_counter = 0
        
        # 3. 检测过拟合
        if train_acc - val_acc > TrainConfig.OVERFIT_THRESHOLD:
            if (self.current_epoch >= 20 and 
                self.current_weight_decay < 1e-3):
                # 过拟合，增加正则化
                old_wd = self.update_weight_decay(self.current_weight_decay * 2)
                guidance_msg += (f" [A优化: 增加正则化 "
                               f"{old_wd:.6f} → {self.current_weight_decay:.6f}]")
        
        # 4. 检测欠拟合
        if self.current_epoch >= 10 and val_acc < 0.97:
            if train_acc < 0.99 and self.current_lr < 0.01:
                # 欠拟合，提高学习率
                old_lr = self.update_learning_rate(self.current_lr * 1.5)
                guidance_msg += (f" [A优化: 提高学习率 "
                               f"{old_lr:.6f} → {self.current_lr:.6f}]")
        
        # 5. 接近目标时的精细调整
        if val_acc >= 0.985 and val_acc < self.target_acc:
            if self.current_lr > TrainConfig.LR_FINE_TUNE:
                old_lr = self.update_learning_rate(TrainConfig.LR_FINE_TUNE)
                guidance_msg = (f" [A优化: 精细调整阶段，"
                              f"lr={self.current_lr:.6f}]")
        
        return guidance_msg
    
    def train(self):
        """
        完整的训练流程
        
        Returns:
            tuple: (final_model, best_val_acc) 训练后的模型和最佳验证准确率
        """
        print_separator("开始反向传播训练")
        print(f"目标准确率: {format_accuracy(self.target_acc)}")
        print(f"最大训练轮数: {self.max_epochs}")
        print(f"初始学习率: {self.current_lr}")
        print("A将持续监控并优化训练策略")
        print("="*DisplayConfig.SEPARATOR_LENGTH)
        
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            
            # 训练一个epoch
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_acc = self.validate()
            self.val_history.append(val_acc)
            
            # A的优化指导
            guidance_msg = self.check_and_optimize(train_acc, val_acc)
            
            # 显示训练进度
            print(DisplayConfig.EPOCH_FORMAT.format(
                epoch=epoch+1,
                train_acc=train_acc,
                val_acc=val_acc,
                lr=self.current_lr,
                msg=guidance_msg
            ))
            
            # 检查是否达到目标
            if val_acc >= self.target_acc:
                print(f"\n{'*'*70}")
                print(f"已达到目标精度 {format_accuracy(self.target_acc)}！")
                print(f"{'*'*70}")
                break
            
            # 早停检查
            if self.patience_counter >= TrainConfig.PATIENCE:
                print(f"\n验证集精度连续 {TrainConfig.PATIENCE} 轮未提升")
                print("A决定停止训练（早停机制）")
                break
        
        # 训练总结
        print_separator("训练完成")
        print(f"最终轮数: {self.current_epoch + 1}")
        print(f"最佳验证准确率: {format_accuracy(self.best_val_acc)}")
        print(f"A共进行了 {self.current_epoch + 1} 轮优化指导")
        print("="*DisplayConfig.SEPARATOR_LENGTH)
        
        return self.model, self.best_val_acc


def train_with_backprop(model, train_loader, val_loader, 
                       target_acc=None, max_epochs=None):
    """
    使用反向传播训练模型（便捷函数）
    
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        target_acc (float, optional): 目标准确率
        max_epochs (int, optional): 最大训练轮数
    
    Returns:
        tuple: (trained_model, final_val_acc)
    """
    trainer = AdaptiveTrainer(model, train_loader, val_loader, 
                             target_acc, max_epochs)
    return trainer.train()


if __name__ == "__main__":
    """
    测试训练器模块
    """
    print("测试训练器模块...")
    
    from models import RandomCNN, FineTuneCNN
    from data_loader import get_mnist_dataloaders
    
    # 加载数据（使用较小的数据集测试）
    print("\n加载数据...")
    train_loader, val_loader, test_loader = get_mnist_dataloaders(
        train_size=1000,  # 只用1000张测试
        val_size=200
    )
    
    # 创建模型
    print("\n创建模型...")
    arch = {
        "n_filters": 32,
        "n_conv_layers": 2,
        "use_pooling": True
    }
    base_cnn = RandomCNN(arch).to(DEVICE)
    model = FineTuneCNN(base_cnn).to(DEVICE)
    
    # 训练（只训练3轮测试）
    print("\n开始训练（3轮测试）...")
    trained_model, val_acc = train_with_backprop(
        model, train_loader, val_loader,
        target_acc=0.90,  # 较低的目标，方便测试
        max_epochs=3
    )
    
    print(f"\n✓ 训练器模块测试完成！最终验证准确率: {format_accuracy(val_acc)}")
