import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import random
import numpy as np


# ========================
# 一些基础设置
# ========================

# 自动选择最佳设备：优先GPU，其次MPS，最后CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"Using device: CUDA (GPU: {torch.cuda.get_device_name(0)})")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = "mps"
    print("Using device: MPS (Apple Silicon)")
else:
    DEVICE = "cpu"
    print("Using device: CPU")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ========================
# 数据加载（只用一小部分，加快 demo）
# ========================

def load_mnist(train_augment=True, train_size=30000, val_size=5000):
    """
    加载 MNIST 数据集，使用数据增强提升性能
    增加训练数据量以达到99%精度
    """
    # 训练集使用数据增强
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 验证集和测试集不增强
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset_raw = datasets.MNIST(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=test_transform
    )

    # 使用更多训练数据
    train_dataset, _ = torch.utils.data.random_split(
        train_dataset_raw, [train_size + val_size, len(train_dataset_raw) - train_size - val_size]
    )
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    return train_dataset, val_dataset, test_dataset


# ========================
# CNN特征提取器（随机权重，不训练）+ 闭式解分类器
# ========================

class RandomCNN(nn.Module):
    """
    随机初始化的CNN，权重固定不训练，只用来提取特征
    """
    def __init__(self, arch):
        super(RandomCNN, self).__init__()
        n_filters = arch.get("n_filters", 32)
        n_conv_layers = arch.get("n_conv_layers", 2)
        use_pooling = arch.get("use_pooling", True)
        
        self.conv_layers = nn.ModuleList()
        self.use_pooling = use_pooling
        
        in_channels = 1
        for i in range(n_conv_layers):
            out_channels = n_filters * (2 ** i)
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            # He初始化
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            nn.init.zeros_(conv.bias)
            self.conv_layers.append(conv)
            in_channels = out_channels
        
        if use_pooling:
            self.pool = nn.MaxPool2d(2, 2)
        
        # 计算输出特征维度
        self.feature_dim = self._get_feature_dim()
    
    def _get_feature_dim(self):
        """计算展平后的特征维度"""
        with torch.no_grad():
            x = torch.zeros(1, 1, 28, 28)
            x = self.forward_features(x)
            return x.view(1, -1).shape[1]
    
    def forward_features(self, x):
        """前向传播，提取特征"""
        for conv in self.conv_layers:
            x = F.relu(conv(x))
            if self.use_pooling:
                x = self.pool(x)
        return x
    
    def forward(self, x):
        """提取特征并展平"""
        x = self.forward_features(x)
        x = x.view(x.size(0), -1)
        return x


def extract_features_with_cnn(X, cnn_model, batch_size=256):
    """
    使用随机CNN提取特征（批量处理避免内存溢出）
    X: (N, 1, 28, 28) 图像张量
    返回: (N, feature_dim) 特征张量
    """
    cnn_model.eval()
    all_features = []
    with torch.no_grad():
        for i in range(0, X.size(0), batch_size):
            batch = X[i:i+batch_size]
            features = cnn_model(batch)
            all_features.append(features.cpu())  # 移到CPU节省GPU内存
    return torch.cat(all_features, dim=0)


def train_B_with_closed_form(X_images, Y, arch, reg=1e-4):
    """
    给定输入图像 X_images (N, 1, 28, 28)、标签 Y (N,), 以及架构 arch,
    用随机CNN提取特征后，用 ridge 回归闭式解训练分类器 W2。
    
    不使用反向传播，CNN权重固定随机。
    返回: (cnn, W2, projection_matrices) 其中projection_matrices包含降维和额外投影矩阵
    """
    device = X_images.device
    # 1) 创建随机CNN特征提取器
    cnn = RandomCNN(arch).to(device)
    projection_matrices = {}
    
    # 2) 提取特征（批量处理，结果在CPU上）
    H = extract_features_with_cnn(X_images, cnn, batch_size=256)  # (N, feature_dim)
    feature_dim = H.shape[1]
    
    # 限制特征维度，避免内存溢出
    MAX_FEATURE_DIM = 4096
    if feature_dim > MAX_FEATURE_DIM:
        # 使用随机投影降维（在CPU上计算）
        std = np.sqrt(2.0 / feature_dim)
        W_reduce = torch.randn(feature_dim, MAX_FEATURE_DIM, dtype=H.dtype) * std
        projection_matrices['W_reduce'] = W_reduce
        H = H @ W_reduce
        feature_dim = MAX_FEATURE_DIM
    
    # 3) 可选：在特征上加一层随机投影
    if arch.get("extra_projection", False):
        proj_dim = min(arch.get("projection_dim", 1024), MAX_FEATURE_DIM)
        std = np.sqrt(2.0 / feature_dim)
        W_proj = torch.randn(feature_dim, proj_dim, dtype=H.dtype) * std
        projection_matrices['W_proj'] = W_proj
        H = F.relu(H @ W_proj)
        feature_dim = proj_dim
    
    # 4) Dropout（训练时随机置零）
    dropout_p = arch.get("dropout_p", 0.0)
    if dropout_p > 0:
        mask = (torch.rand(H.shape) > dropout_p).float()
        H = H * mask / (1.0 - dropout_p)
    
    # 5) 闭式解：W2 = (H^T H + reg*I)^-1 H^T Y_onehot（在CPU上计算）
    Y_cpu = Y.cpu()
    Y_onehot = F.one_hot(Y_cpu, num_classes=10).float()
    HtH = H.T @ H
    HtY = H.T @ Y_onehot
    I = torch.eye(feature_dim, dtype=H.dtype)  # CPU上的单位矩阵
    
    W2 = torch.linalg.solve(HtH + reg * I, HtY)
    
    return cnn, W2, projection_matrices


def evaluate_B(X_images, Y, arch, cnn, W2, projection_matrices):
    """
    给定图像 X_images, 标签 Y, 架构 arch, CNN模型、分类器 W2和投影矩阵，计算准确率。
    不需要反向传播，纯前向。
    使用训练时保存的投影矩阵，保证特征一致性。
    """
    # 提取特征（批量处理，返回CPU上的张量）
    H = extract_features_with_cnn(X_images, cnn, batch_size=256)
    
    # 应用训练时的降维矩阵（如果有）
    if 'W_reduce' in projection_matrices:
        H = H @ projection_matrices['W_reduce']
    
    # 应用训练时的额外投影矩阵（如果有）
    if 'W_proj' in projection_matrices:
        H = F.relu(H @ projection_matrices['W_proj'])
    
    logits = H @ W2
    preds = logits.argmax(dim=1)
    Y_cpu = Y.cpu()
    acc = (preds == Y_cpu).float().mean().item()
    return acc


# ========================
# 架构进化（A 的作用）
# ========================

def sample_arch():
    """
    随机采样CNN架构（卷积滤波器数量、卷积层数、是否池化、额外投影等）
    限制：当use_pooling=False时，限制层数为2，避免内存溢出
    """
    n_filters = random.choice([16, 32, 64])
    use_pooling = random.choice([True, False])
    
    # 不池化时只用2层，池化时可以用2-3层
    if use_pooling:
        n_conv_layers = random.choice([2, 3])
    else:
        n_conv_layers = 2
    
    dropout_p = random.choice([0.0, 0.1, 0.2, 0.3])
    extra_projection = random.choice([False, True])
    projection_dim = random.choice([512, 1024, 2048]) if extra_projection else 512
    
    return {
        "n_filters": n_filters,
        "n_conv_layers": n_conv_layers,
        "use_pooling": use_pooling,
        "dropout_p": dropout_p,
        "extra_projection": extra_projection,
        "projection_dim": projection_dim
    }


def mutate_arch(arch):
    """
    对架构做突变
    """
    new_arch = arch.copy()
    param = random.choice(["n_filters", "n_conv_layers", "use_pooling", "dropout_p", "extra_projection"])
    
    if param == "n_filters":
        choices = [16, 32, 64]
        idx = choices.index(arch["n_filters"])
        new_idx = min(max(idx + random.choice([-1, 0, 1]), 0), len(choices) - 1)
        new_arch["n_filters"] = choices[new_idx]
    elif param == "n_conv_layers":
        # 不池化时固定2层，池化时可以2-3层
        if new_arch["use_pooling"]:
            new_arch["n_conv_layers"] = max(2, min(3, arch["n_conv_layers"] + random.choice([-1, 0, 1])))
        else:
            new_arch["n_conv_layers"] = 2
    elif param == "use_pooling":
        new_arch["use_pooling"] = not arch["use_pooling"]
        # 切换池化状态时，调整层数
        if not new_arch["use_pooling"]:
            new_arch["n_conv_layers"] = 2  # 不池化时固定2层
    elif param == "dropout_p":
        choices = [0.0, 0.1, 0.2, 0.3]
        idx = choices.index(arch["dropout_p"])
        new_idx = min(max(idx + random.choice([-1, 0, 1]), 0), len(choices) - 1)
        new_arch["dropout_p"] = choices[new_idx]
    else:  # extra_projection
        new_arch["extra_projection"] = not arch["extra_projection"]
        if new_arch["extra_projection"]:
            new_arch["projection_dim"] = random.choice([512, 1024, 2048])
    
    return new_arch


def evolution_search(
    train_loader, val_loader,
    pop_size=12, n_generations=8, reg=1e-4, target_acc=0.93
):
    """
    进化搜索：
      - 每个架构 A，用随机CNN提取特征后，用闭式解训练分类器
      - 根据验证集 acc 淘汰并繁殖
      - CNN权重固定随机，不使用反向传播
      - 当达到target_acc时，提前结束进化
    """
    population = [sample_arch() for _ in range(pop_size)]
    top_k = max(2, pop_size // 3)

    best_arch = None
    best_acc = 0.0
    best_cnn = None
    best_W2 = None
    best_proj_mats = {}

    for gen in range(n_generations):
        print(f"\n=== Generation {gen} ===")
        scored = []

        # 把训练集和验证集加载成张量（保持图像格式）
        X_train_list, Y_train_list = [], []
        for (imgs, labels) in train_loader:
            X_train_list.append(imgs.to(DEVICE))
            Y_train_list.append(labels.to(DEVICE))
        X_train = torch.cat(X_train_list, dim=0)
        Y_train = torch.cat(Y_train_list, dim=0)

        X_val_list, Y_val_list = [], []
        for (imgs, labels) in val_loader:
            X_val_list.append(imgs.to(DEVICE))
            Y_val_list.append(labels.to(DEVICE))
        X_val = torch.cat(X_val_list, dim=0)
        Y_val = torch.cat(Y_val_list, dim=0)

        for i, arch in enumerate(population):
            cnn, W2, proj_mats = train_B_with_closed_form(X_train, Y_train, arch, reg=reg)
            val_acc = evaluate_B(X_val, Y_val, arch, cnn, W2, proj_mats)
            scored.append((val_acc, arch, cnn, W2, proj_mats))
            print(f"  Individual {i}: arch={arch}, val_acc={val_acc:.4f}")

        # 选出本代最优
        scored.sort(key=lambda x: x[0], reverse=True)
        best_gen_acc, best_gen_arch, best_gen_cnn, best_gen_W2, best_gen_proj = scored[0]
        print(
            f"  >> Best in gen {gen}: arch={best_gen_arch}, "
            f"val_acc={best_gen_acc:.4f}"
        )

        if best_gen_acc > best_acc:
            best_acc = best_gen_acc
            best_arch = best_gen_arch
            best_cnn = best_gen_cnn
            best_W2 = best_gen_W2
            best_proj_mats = best_gen_proj

        # 如果达到目标精度，提前结束进化
        if best_acc >= target_acc:
            print(f"\n*** Reached target accuracy {target_acc:.2%}, stopping evolution ***")
            break

        # 选择 + 繁殖
        elites = [arch for _, arch, _, _, _ in scored[:top_k]]
        new_population = elites.copy()
        while len(new_population) < pop_size:
            parent = random.choice(elites)
            child = mutate_arch(parent)
            new_population.append(child)

        population = new_population

    print("\n=== Evolution finished ===")
    print(f"Best arch overall: {best_arch}, val_acc={best_acc:.4f}")
    return best_arch, best_cnn, best_W2, best_proj_mats


# ========================
# 反向传播微调（当闭式解达到93%后使用）
# ========================

class FineTuneCNN(nn.Module):
    """
    可训练的CNN分类器，用于微调
    使用更深的分类器以达到99%精度
    """
    def __init__(self, base_cnn, num_classes=10):
        super(FineTuneCNN, self).__init__()
        self.base_cnn = base_cnn
        # 计算特征维度
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28).to(next(base_cnn.parameters()).device)
            feat_dim = base_cnn(dummy).shape[1]
        
        # 添加更深的可训练分类层
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # 提取特征
        features = self.base_cnn(x)
        # 分类
        logits = self.classifier(features)
        return logits


def finetune_with_backprop(model, train_loader, val_loader, target_acc=0.991, max_epochs=100):
    """
    使用反向传播微调模型，直至达到target_acc
    在此过程中，A持续监控训练状态并动态优化训练策略
    """
    print(f"\n{'='*60}")
    print("开始使用反向传播微调模型 (目标: {:.2%})".format(target_acc))
    print("A将持续监控并优化训练策略")
    print('='*60)
    
    # A的初始训练策略 - 使用更好的优化器和损失函数
    current_lr = 0.001
    current_weight_decay = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=current_lr, weight_decay=current_weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing防止过拟合
    
    best_val_acc = 0.0
    patience_counter = 0
    stagnation_counter = 0
    max_patience = 15
    
    # A的优化历史记录
    val_history = []
    
    for epoch in range(max_epochs):
        # 训练
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = train_correct / train_total
        
        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = val_correct / val_total
        val_history.append(val_acc)
        
        # === A对B的优化指导 ===
        guidance_msg = ""
        
        # 1. 动态学习率调整
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            stagnation_counter = 0
            
            # 达到目标精度
            if val_acc >= target_acc:
                print(f"Epoch {epoch+1:3d}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, lr={current_lr:.6f}")
                print(f"\n*** 达到目标精度 {target_acc:.2%}! ***")
                return model, val_acc
        else:
            patience_counter += 1
            stagnation_counter += 1
        
        # 2. A检测训练停滞并调整策略
        if stagnation_counter >= 4:  # 更快响应
            # 检查最近几个epoch的进步
            if len(val_history) >= 5:
                recent_improvement = val_history[-1] - val_history[-5]
                
                if recent_improvement < 0.001:  # 进步小于0.1%
                    # A决定降低学习率
                    old_lr = current_lr
                    current_lr *= 0.5
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr
                    guidance_msg = f" [A优化: 降低学习率 {old_lr:.6f} → {current_lr:.6f}]"
                    stagnation_counter = 0
        
        # 3. A检测过拟合并调整正则化
        if train_acc - val_acc > 0.02:  # 训练集比验证集高2%以上
            if epoch >= 20 and current_weight_decay < 1e-3:
                old_wd = current_weight_decay
                current_weight_decay *= 2
                # 重新创建优化器以更新weight_decay
                optimizer = torch.optim.Adam(model.parameters(), lr=current_lr, weight_decay=current_weight_decay)
                guidance_msg += f" [A优化: 增加正则化 {old_wd:.6f} → {current_weight_decay:.6f}]"
        
        # 4. A检测欠拟合并调整策略
        if epoch >= 10 and val_acc < 0.97:
            if train_acc < 0.99:  # 训练集也不够好
                if current_lr < 0.01:
                    old_lr = current_lr
                    current_lr *= 1.5
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr
                    guidance_msg += f" [A优化: 提高学习率 {old_lr:.6f} → {current_lr:.6f}]"
        
        # 5. A在接近目标时精细调整
        if val_acc >= 0.985 and val_acc < target_acc:
            if current_lr > 0.00005:  # 更小的学习率
                old_lr = current_lr
                current_lr = 0.00005
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                guidance_msg = f" [A优化: 精细调整阶段，lr={current_lr:.6f}]"
        
        print(f"Epoch {epoch+1:3d}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, lr={current_lr:.6f}{guidance_msg}")
        
        # 早停
        if patience_counter >= max_patience:
            print(f"\n验证集精度 {max_patience} 个epoch未提升，A决定停止训练")
            break
    
    print(f"\n微调完成，最佳验证集精度: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"A共进行了 {epoch+1} 轮优化指导")
    return model, best_val_acc


# ========================
# 主函数
# ========================

def main():
    train_dataset, val_dataset, test_dataset = load_mnist()

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=0
    )

    print(f"Train set: {len(train_dataset)} samples (使用数据增强)")
    print(f"Val set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")

    # 阶段1: 使用进化搜索 + 闭式解达到93%
    print("\n" + "="*70)
    print("阶段1: 进化搜索 + 闭式解 (目标: 93%)")
    print("="*70)
    
    best_arch, best_cnn, best_W2, best_proj_mats = evolution_search(
        train_loader, val_loader,
        pop_size=10, n_generations=15, reg=1e-4, target_acc=0.93
    )

    # 阶段1验证
    X_val_list, Y_val_list = [], []
    for (imgs, labels) in val_loader:
        X_val_list.append(imgs.to(DEVICE))
        Y_val_list.append(labels.to(DEVICE))
    X_val = torch.cat(X_val_list, dim=0)
    Y_val = torch.cat(Y_val_list, dim=0)
    
    phase1_acc = evaluate_B(X_val, Y_val, best_arch, best_cnn, best_W2, best_proj_mats)
    print(f"\n阶段1完成，验证集精度: {phase1_acc:.4f} ({phase1_acc*100:.2f}%)")
    
    # 阶段2: 固定架构B，使用反向传播继续训练至99.1%
    if phase1_acc >= 0.90:  # 只有达到90%以上才值得继续训练
        print("\n" + "="*70)
        print("阶段2: 固定架构B，使用反向传播训练 (目标: 99.1%)")
        print("="*70)
        print(f"固定的架构: {best_arch}")
        
        # 创建可训练模型（基于阶段1的CNN特征提取器）
        finetune_model = FineTuneCNN(best_cnn, num_classes=10).to(DEVICE)
        
        # 使用反向传播微调
        finetune_model, final_val_acc = finetune_with_backprop(
            finetune_model, train_loader, val_loader, 
            target_acc=0.991, max_epochs=150
        )
        
        # 最终测试
        finetune_model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = finetune_model(imgs)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        final_test_acc = test_correct / test_total
        
        print("\n" + "="*70)
        print("训练完成!")
        print("="*70)
        print(f"阶段1 (闭式解) 验证集精度: {phase1_acc:.4f} ({phase1_acc*100:.2f}%)")
        print(f"阶段2 (反向传播) 验证集精度: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
        print(f"最终测试集精度: {final_test_acc:.4f} ({final_test_acc*100:.2f}%)")
        print(f"最佳架构: {best_arch}")
    else:
        print(f"\n阶段1精度未达到90%，不进行阶段2训练")
        print(f"当前精度: {phase1_acc:.4f} ({phase1_acc*100:.2f}%)")


if __name__ == "__main__":
    main()
