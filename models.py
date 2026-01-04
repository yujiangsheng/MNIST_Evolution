"""
模型模块 - 定义神经网络架构

该模块包含两个主要模型：
1. RandomCNN: 随机初始化的卷积神经网络，权重固定不训练，仅用于特征提取
2. FineTuneCNN: 可训练的完整CNN分类器，用于反向传播微调

设计思路：
- 阶段1使用RandomCNN提取特征，配合闭式解训练分类器
- 阶段2使用FineTuneCNN进行端到端的反向传播训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig


class RandomCNN(nn.Module):
    """
    随机权重的CNN特征提取器
    
    这个模型的特点：
    1. 权重随机初始化后固定不变，不参与反向传播训练
    2. 使用He初始化保证特征提取的有效性
    3. 只负责将输入图像转换为高维特征向量
    
    理论基础：
    - 随机特征理论：即使权重是随机的，深度CNN也能提取有用的特征
    - 适用于快速架构搜索，避免每次都训练网络
    
    架构参数：
    - n_filters: 第一层卷积核数量（后续层按2倍递增）
    - n_conv_layers: 卷积层数量（2或3）
    - use_pooling: 是否在每个卷积层后使用最大池化
    """
    
    def __init__(self, arch):
        """
        初始化随机CNN
        
        Args:
            arch (dict): 架构配置字典，包含：
                - n_filters (int): 卷积核数量
                - n_conv_layers (int): 卷积层数
                - use_pooling (bool): 是否使用池化
        """
        super(RandomCNN, self).__init__()
        
        # 从架构配置中提取参数
        n_filters = arch.get("n_filters", 32)
        n_conv_layers = arch.get("n_conv_layers", 2)
        use_pooling = arch.get("use_pooling", True)
        
        # 存储配置
        self.use_pooling = use_pooling
        self.conv_layers = nn.ModuleList()
        
        # 构建卷积层
        in_channels = 1  # MNIST是灰度图，输入通道数为1
        for i in range(n_conv_layers):
            # 每一层的卷积核数量翻倍：n_filters, 2*n_filters, 4*n_filters, ...
            out_channels = n_filters * (2 ** i)
            
            # 创建卷积层
            # kernel_size=3: 3x3卷积核，能捕获局部模式
            # padding=1: 保持特征图尺寸不变（对于3x3卷积）
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            
            # He初始化（也称为Kaiming初始化）
            # 专门为ReLU激活函数设计，能保持信号方差在前向传播中稳定
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            nn.init.zeros_(conv.bias)  # 偏置初始化为0
            
            self.conv_layers.append(conv)
            in_channels = out_channels  # 下一层的输入通道数
        
        # 池化层（如果使用）
        if use_pooling:
            # 2x2最大池化，stride=2，将特征图尺寸减半
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 计算最终的特征维度（用于后续的全连接层）
        self.feature_dim = self._get_feature_dim()
    
    def _get_feature_dim(self):
        """
        计算网络输出的特征维度
        
        通过前向传播一个dummy输入来自动计算特征维度
        这样就不需要手动计算复杂的卷积和池化后的尺寸
        
        Returns:
            int: 展平后的特征维度
        """
        with torch.no_grad():  # 不需要计算梯度
            # 创建一个假的输入：1张28x28的图像
            dummy_input = torch.zeros(1, 1, 28, 28)
            # 前向传播
            features = self.forward_features(dummy_input)
            # 展平并返回维度
            return features.view(1, -1).shape[1]
    
    def forward_features(self, x):
        """
        提取卷积特征（不展平）
        
        Args:
            x (torch.Tensor): 输入图像 (batch_size, 1, 28, 28)
        
        Returns:
            torch.Tensor: 卷积特征 (batch_size, channels, height, width)
        """
        # 逐层应用卷积和激活
        for conv in self.conv_layers:
            x = conv(x)          # 卷积操作
            x = F.relu(x)        # ReLU激活函数（引入非线性）
            
            if self.use_pooling:
                x = self.pool(x)  # 最大池化（降低空间分辨率）
        
        return x
    
    def forward(self, x):
        """
        完整的前向传播：提取特征并展平
        
        Args:
            x (torch.Tensor): 输入图像 (batch_size, 1, 28, 28)
        
        Returns:
            torch.Tensor: 展平的特征向量 (batch_size, feature_dim)
        """
        x = self.forward_features(x)  # 提取卷积特征
        x = x.view(x.size(0), -1)     # 展平：(batch, C, H, W) -> (batch, C*H*W)
        return x


class FineTuneCNN(nn.Module):
    """
    可训练的完整CNN分类器（用于反向传播微调）
    
    这个模型包含两部分：
    1. 特征提取器：基于RandomCNN的卷积层
    2. 分类器：多层全连接网络
    
    设计要点：
    - 使用BatchNorm加速训练并提升稳定性
    - 使用Dropout防止过拟合
    - 分类器较深，能够学习复杂的决策边界
    
    用于阶段2的反向传播训练，目标是达到99%以上的准确率
    """
    
    def __init__(self, base_cnn, num_classes=ModelConfig.NUM_CLASSES):
        """
        初始化微调模型
        
        Args:
            base_cnn (RandomCNN): 预训练的特征提取器（可以是随机初始化的）
            num_classes (int): 分类类别数（MNIST为10）
        """
        super(FineTuneCNN, self).__init__()
        
        # 特征提取器（来自RandomCNN）
        self.base_cnn = base_cnn
        
        # 计算特征维度
        with torch.no_grad():
            # 创建dummy输入来获取特征维度
            device = next(base_cnn.parameters()).device
            dummy = torch.zeros(1, 1, 28, 28).to(device)
            feat_dim = base_cnn(dummy).shape[1]
        
        # 构建分类器
        # 使用Sequential容器按顺序组织各层
        hidden_dims = ModelConfig.CLASSIFIER_HIDDEN_DIMS
        dropout_rates = ModelConfig.CLASSIFIER_DROPOUT
        
        self.classifier = nn.Sequential(
            # 第一个全连接层：feat_dim -> 512
            nn.Linear(feat_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),  # 批归一化
            nn.ReLU(),                        # ReLU激活
            nn.Dropout(dropout_rates[0]),     # Dropout防止过拟合
            
            # 第二个全连接层：512 -> 256
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rates[1]),
            
            # 输出层：256 -> num_classes
            nn.Linear(hidden_dims[1], num_classes)
            # 注意：输出层不使用激活函数，因为CrossEntropyLoss内部包含Softmax
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入图像 (batch_size, 1, 28, 28)
        
        Returns:
            torch.Tensor: 类别logits (batch_size, num_classes)
        """
        # 1. 提取特征
        features = self.base_cnn(x)  # (batch_size, feat_dim)
        
        # 2. 分类
        logits = self.classifier(features)  # (batch_size, num_classes)
        
        return logits


class EnsembleModel(nn.Module):
    """
    集成模型（可选，用于进一步提升性能）
    
    通过集成多个模型的预测来提升准确率和鲁棒性
    目前暂未使用，预留用于未来优化
    """
    
    def __init__(self, models):
        """
        初始化集成模型
        
        Args:
            models (list): 要集成的模型列表
        """
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, x):
        """
        集成前向传播：平均所有模型的预测
        
        Args:
            x (torch.Tensor): 输入图像
        
        Returns:
            torch.Tensor: 平均后的logits
        """
        # 获取所有模型的预测
        outputs = [model(x) for model in self.models]
        
        # 平均预测结果
        avg_output = torch.stack(outputs).mean(dim=0)
        
        return avg_output


def count_parameters(model):
    """
    统计模型的参数数量
    
    Args:
        model (nn.Module): PyTorch模型
    
    Returns:
        tuple: (total_params, trainable_params) 总参数数和可训练参数数
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def print_model_info(model, model_name="Model"):
    """
    打印模型信息
    
    Args:
        model (nn.Module): PyTorch模型
        model_name (str): 模型名称
    """
    total, trainable = count_parameters(model)
    print(f"\n{model_name} 信息:")
    print(f"  总参数数: {total:,}")
    print(f"  可训练参数数: {trainable:,}")
    print(f"  模型结构:\n{model}")


if __name__ == "__main__":
    """
    测试模型模块
    运行: python models.py
    """
    print("测试模型模块...")
    
    # 测试RandomCNN
    print("\n" + "="*60)
    print("测试 RandomCNN")
    print("="*60)
    
    arch = {
        "n_filters": 32,
        "n_conv_layers": 3,
        "use_pooling": True
    }
    
    random_cnn = RandomCNN(arch)
    print_model_info(random_cnn, "RandomCNN")
    
    # 测试前向传播
    dummy_input = torch.randn(4, 1, 28, 28)  # 4张图像
    features = random_cnn(dummy_input)
    print(f"\n输入形状: {dummy_input.shape}")
    print(f"输出特征形状: {features.shape}")
    print(f"特征维度: {random_cnn.feature_dim}")
    
    # 测试FineTuneCNN
    print("\n" + "="*60)
    print("测试 FineTuneCNN")
    print("="*60)
    
    finetune_cnn = FineTuneCNN(random_cnn)
    print_model_info(finetune_cnn, "FineTuneCNN")
    
    # 测试前向传播
    logits = finetune_cnn(dummy_input)
    print(f"\n输入形状: {dummy_input.shape}")
    print(f"输出logits形状: {logits.shape}")
    
    # 测试预测
    predictions = logits.argmax(dim=1)
    print(f"预测类别: {predictions}")
    
    print("\n✓ 模型模块测试通过！")
