"""
快速参考指南 - 常用代码示例

这个文件包含了项目中常用功能的示例代码，方便快速参考和使用。
"""

# ============================================================================
# 1. 基础使用
# ============================================================================

# 1.1 运行完整训练流程
"""
python main.py
"""

# 1.2 测试单个模块
"""
python data_loader.py  # 测试数据加载
python models.py       # 测试模型
python evolution.py    # 测试进化算法
python trainer.py      # 测试训练器
python utils.py        # 测试工具函数
"""


# ============================================================================
# 2. 自定义配置
# ============================================================================

# 2.1 修改训练参数
"""
# 编辑 config.py 文件

class TrainConfig:
    PHASE1_TARGET_ACC = 0.95   # 提高阶段1目标
    PHASE2_TARGET_ACC = 0.995  # 提高阶段2目标
    MAX_EPOCHS = 200           # 增加最大训练轮数
    INITIAL_LR = 0.0005        # 降低初始学习率
"""

# 2.2 修改数据集大小
"""
class DataConfig:
    TRAIN_SIZE = 50000  # 使用更多训练数据
    VAL_SIZE = 10000    # 使用更多验证数据
"""

# 2.3 修改进化参数
"""
class EvolutionConfig:
    POPULATION_SIZE = 20    # 增大种群
    MAX_GENERATIONS = 20    # 增加进化代数
"""


# ============================================================================
# 3. 加载和使用数据
# ============================================================================

# 3.1 加载数据集
from data_loader import get_mnist_dataloaders

train_loader, val_loader, test_loader = get_mnist_dataloaders()

# 3.2 自定义数据集大小
train_loader, val_loader, test_loader = get_mnist_dataloaders(
    train_size=10000,  # 使用10000张训练
    val_size=2000      # 使用2000张验证
)

# 3.3 获取一个批次数据
images, labels = next(iter(train_loader))
print(f"批次大小: {images.shape[0]}")
print(f"图像形状: {images.shape}")  # (batch_size, 1, 28, 28)
print(f"标签形状: {labels.shape}")  # (batch_size,)


# ============================================================================
# 4. 创建和使用模型
# ============================================================================

# 4.1 创建随机CNN
from models import RandomCNN
from config import DEVICE

arch = {
    "n_filters": 64,
    "n_conv_layers": 3,
    "use_pooling": True
}

cnn = RandomCNN(arch).to(DEVICE)
print(f"特征维度: {cnn.feature_dim}")

# 4.2 提取特征
import torch
dummy_input = torch.randn(4, 1, 28, 28).to(DEVICE)
features = cnn(dummy_input)
print(f"特征形状: {features.shape}")  # (4, feature_dim)

# 4.3 创建可训练模型
from models import FineTuneCNN

finetune_model = FineTuneCNN(cnn, num_classes=10).to(DEVICE)

# 4.4 前向传播
logits = finetune_model(dummy_input)
predictions = logits.argmax(dim=1)
print(f"预测结果: {predictions}")


# ============================================================================
# 5. 进化算法使用
# ============================================================================

# 5.1 随机采样架构
from evolution import sample_architecture

arch = sample_architecture()
print(f"随机架构: {arch}")

# 5.2 架构突变
from evolution import mutate_architecture

mutated = mutate_architecture(arch)
print(f"突变后: {mutated}")

# 5.3 运行进化搜索
from evolution import evolution_search

best_arch, best_cnn, best_W2, best_proj = evolution_search(
    train_loader, val_loader,
    pop_size=5,        # 小种群快速测试
    n_generations=3,   # 少代数快速测试
    target_acc=0.90    # 较低目标快速测试
)


# ============================================================================
# 6. 训练模型
# ============================================================================

# 6.1 使用自适应训练器
from trainer import AdaptiveTrainer

trainer = AdaptiveTrainer(
    finetune_model, 
    train_loader, 
    val_loader,
    target_acc=0.99,
    max_epochs=50
)

trained_model, final_acc = trainer.train()

# 6.2 使用便捷函数训练
from trainer import train_with_backprop

trained_model, final_acc = train_with_backprop(
    finetune_model, 
    train_loader, 
    val_loader,
    target_acc=0.99,
    max_epochs=50
)


# ============================================================================
# 7. 评估模型
# ============================================================================

# 7.1 计算准确率
from utils import evaluate_accuracy

test_acc = evaluate_accuracy(trained_model, test_loader, DEVICE)
print(f"测试集准确率: {test_acc:.4f}")

# 7.2 计算损失
import torch.nn as nn
from utils import evaluate_loss

criterion = nn.CrossEntropyLoss()
test_loss = evaluate_loss(trained_model, test_loader, criterion, DEVICE)
print(f"测试集损失: {test_loss:.4f}")

# 7.3 计算混淆矩阵
from utils import compute_confusion_matrix

confusion = compute_confusion_matrix(
    trained_model, test_loader, 
    num_classes=10, device=DEVICE
)
print(f"混淆矩阵:\n{confusion}")


# ============================================================================
# 8. 保存和加载模型
# ============================================================================

# 8.1 保存模型
from utils import save_model

save_model(
    trained_model, 
    'best_model.pth',
    additional_info={
        'architecture': best_arch,
        'test_accuracy': test_acc,
        'epoch': 50
    }
)

# 8.2 加载模型
from utils import load_model

# 先创建模型结构
loaded_model = FineTuneCNN(cnn, num_classes=10)

# 加载权重
loaded_model, extra_info = load_model(
    loaded_model, 
    'best_model.pth', 
    DEVICE
)

print(f"加载的模型信息: {extra_info}")


# ============================================================================
# 9. 工具函数使用
# ============================================================================

# 9.1 格式化输出
from utils import format_accuracy, print_separator, print_architecture

print_separator("开始训练")
acc = 0.9543
print(f"准确率: {format_accuracy(acc)}")  # 0.9543 (95.43%)

# 9.2 打印架构
print_architecture(best_arch, "最优架构")

# 9.3 设置随机种子
from utils import set_seed

set_seed(42)  # 确保可重复性

# 9.4 创建随机投影矩阵
from utils import create_random_projection

proj_matrix = create_random_projection(
    input_dim=1000, 
    output_dim=100, 
    initialization='he'
)


# ============================================================================
# 10. 完整训练示例
# ============================================================================

def custom_training_example():
    """自定义训练流程示例"""
    
    # 1. 加载数据
    from data_loader import get_mnist_dataloaders
    train_loader, val_loader, test_loader = get_mnist_dataloaders(
        train_size=20000,
        val_size=5000
    )
    
    # 2. 进化搜索
    from evolution import evolution_search
    best_arch, best_cnn, _, _ = evolution_search(
        train_loader, val_loader,
        pop_size=8,
        n_generations=5,
        target_acc=0.92
    )
    
    # 3. 创建可训练模型
    from models import FineTuneCNN
    from config import DEVICE
    model = FineTuneCNN(best_cnn).to(DEVICE)
    
    # 4. 训练模型
    from trainer import train_with_backprop
    trained_model, val_acc = train_with_backprop(
        model, train_loader, val_loader,
        target_acc=0.98,
        max_epochs=30
    )
    
    # 5. 测试评估
    from utils import evaluate_accuracy
    test_acc = evaluate_accuracy(trained_model, test_loader, DEVICE)
    
    # 6. 保存模型
    from utils import save_model
    save_model(trained_model, 'my_model.pth', {
        'architecture': best_arch,
        'test_accuracy': test_acc
    })
    
    print(f"训练完成！测试准确率: {test_acc:.4f}")
    return trained_model, test_acc


# ============================================================================
# 11. 常见问题解决
# ============================================================================

# 11.1 内存不足
"""
解决方案：
1. 减小批次大小
   DataConfig.BATCH_SIZE_TRAIN = 64  # 默认128
   
2. 减小特征维度限制
   ModelConfig.MAX_FEATURE_DIM = 2048  # 默认4096
   
3. 使用CPU训练（在config.py中修改）
"""

# 11.2 训练速度慢
"""
解决方案：
1. 减少训练数据
   DataConfig.TRAIN_SIZE = 10000
   
2. 减少进化代数
   EvolutionConfig.MAX_GENERATIONS = 10
   
3. 减少种群大小
   EvolutionConfig.POPULATION_SIZE = 5
"""

# 11.3 准确率不够高
"""
解决方案：
1. 增加训练数据
   DataConfig.TRAIN_SIZE = 50000
   
2. 延长训练
   TrainConfig.MAX_EPOCHS = 200
   
3. 开启数据增强
   DataConfig.USE_AUGMENTATION = True
   
4. 调整学习率
   TrainConfig.INITIAL_LR = 0.0005
"""


# ============================================================================
# 12. 进阶使用
# ============================================================================

# 12.1 自定义架构空间
"""
在 evolution.py 中修改 sample_architecture() 函数
添加新的超参数选择
"""

# 12.2 自定义训练策略
"""
在 trainer.py 中修改 AdaptiveTrainer 类
添加新的优化策略
"""

# 12.3 添加新的数据增强
"""
在 data_loader.py 中修改 get_train_transforms() 函数
添加新的变换操作
"""


if __name__ == "__main__":
    """
    运行示例代码
    """
    print("MNIST进化学习 - 快速参考指南")
    print("="*60)
    print("查看此文件获取常用代码示例")
    print("运行 python main.py 开始完整训练")
    print("="*60)
