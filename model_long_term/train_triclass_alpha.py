"""
用新的 Alpha 标签训练三分类模型（改进版）
期望：模型预测概率能有真正的区分度，而不是接近随机的[0.33, 0.33, 0.33]

改进：保存标准化参数（scaler）供回测使用，避免数据泄露
"""
import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

sys.path.append('.')
from model_triclass import GRUModelTriclass, ConvGRUAttentionTriclass
from data_loader_triclass import load_stock_data, get_class_weights, StockSequenceDataset

print("=" * 90)
print("用 Alpha 标签训练三分类模型（改进版）")
print("=" * 90)

# 配置
BATCH_SIZE = 128
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
SPLIT_DATE = '2023-12-31'  # 只用2023年底前的数据
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\n配置信息：")
print(f"  设备: {DEVICE}")
print(f"  训练截止日期: {SPLIT_DATE}")
print(f"  批大小: {BATCH_SIZE}")
print(f"  学习率: {LEARNING_RATE}")
print(f"  训练轮数: {NUM_EPOCHS}")

# 1. 加载数据
print(f"\n1. 加载数据...")
try:
    X, y, dist, feature_cols, global_scaler = load_stock_data(
        data_dir='../features',
        window_size=60,
        predict_days=20,
        split_date=SPLIT_DATE,
        feature_cols=None,
        scaler=None  # 训练时学习新的scaler
    )
except Exception as e:
    print(f"❌ 加载失败: {e}")
    exit(1)

# 保存scaler供回测使用
print(f"\n保存标准化参数...")
with open('scaler_triclass.pkl', 'wb') as f:
    pickle.dump(global_scaler, f)
print(f"✓ Scaler已保存到 scaler_triclass.pkl")

# 2. 分割训练/验证集
print(f"\n2. 分割训练/验证集...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  训练集: {X_train.shape}")
print(f"  验证集: {X_val.shape}")

# 3. 创建数据集和加载器
train_dataset = StockSequenceDataset(X_train, y_train)
val_dataset = StockSequenceDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 4. 初始化模型（Conv1d + GRU + Attention）
print(f"\n3. 初始化改进的模型 (Conv1d + GRU + Attention)...")
model = ConvGRUAttentionTriclass(
    in_feats=X.shape[2],
    conv_channels=64,
    conv_k=5,
    rnn_hidden=128,
    rnn_layers=1,
    attn_dim=64,
    dropout=0.3,
    num_classes=3,
    return_logits_default=False  # 训练时手动要 logits
)
model.to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  总参数数: {total_params:,}")
print(f"  可训练参数: {trainable_params:,}")
print(f"  模型架构:")
print(f"    ✓ Conv1d: 5x1 kernel, channels=64")
print(f"    ✓ GRU隐层: 128, 层数: 1")
print(f"    ✓ Attention 汇聚时间维")
print(f"    ✓ Dropout: 0.3")

# 5. 优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 计算类别权重（平衡不平衡的类）
class_weights = get_class_weights(y_train)
class_weights = class_weights.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)

print(f"  类别权重: {class_weights.tolist()}")
print(f"  损失函数: CrossEntropyLoss (加权)")

# 6. 训练循环
print(f"\n4. 开始训练...\n")
print(f"策略: 至少训练100轮，之后如果100轮内无改善则早停\n")

best_val_loss = float('inf')
patience = 10# 100轮无改善才早停
min_epochs = 50# 至少训练100轮
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    # 训练
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(X_batch, return_logits=True)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        # 计算准确率
        _, pred = torch.max(logits, 1)
        train_correct += (pred == y_batch).sum().item()
        train_total += y_batch.size(0)
    
    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = train_correct / train_total * 100
    
    # 验证
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            logits = model(X_batch, return_logits=True)
            loss = criterion(logits, y_batch)
            
            val_loss += loss.item()
            
            _, pred = torch.max(logits, 1)
            val_correct += (pred == y_batch).sum().item()
            val_total += y_batch.size(0)
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = val_correct / val_total * 100
    
    # 打印进度
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.2f}%")
    
    # 保存最佳模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'model_triclass_alpha.pth')
    else:
        patience_counter += 1
        # 只在达到最小轮数后才考虑早停
        if epoch >= min_epochs and patience_counter >= patience:
            print(f"\n✓ 早停：第 {epoch+1} 轮时触发 ({patience}轮无改善)")
            break

# 7. 测试模型预测能力
print(f"\n5. 测试模型预测能力...\n")

model.load_state_dict(torch.load('model_triclass_alpha.pth', map_location=DEVICE))
model.eval()

# 在验证集上测试预测分布
all_probs = []
with torch.no_grad():
    for X_batch, _ in val_loader:
        X_batch = X_batch.to(DEVICE)
        logits = model(X_batch, return_logits=True)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())

all_probs = np.concatenate(all_probs, axis=0)

print("预测概率分布（验证集）:")
print(f"  P(Weak)   : {all_probs[:, 0].mean():.4f} ± {all_probs[:, 0].std():.4f}")
print(f"  P(Average): {all_probs[:, 1].mean():.4f} ± {all_probs[:, 1].std():.4f}")
print(f"  P(Strong) : {all_probs[:, 2].mean():.4f} ± {all_probs[:, 2].std():.4f}")

# 检查是否有真正的区分度
is_random = (
    abs(all_probs[:, 0].mean() - 0.333) < 0.01 and
    abs(all_probs[:, 1].mean() - 0.333) < 0.01 and
    abs(all_probs[:, 2].mean() - 0.333) < 0.01
)

if is_random:
    print("\n⚠️  警告：模型预测仍接近随机 [0.33, 0.33, 0.33]")
    print("   建议：检查特征、标签或模型架构")
else:
    print("\n✓ 好消息：模型预测有明显的区分度")
    print("   模型已学到有意义的特征")

# 8. 最终测试准确率
final_val_loss = 0
final_val_correct = 0
final_val_total = 0

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        
        logits = model(X_batch, return_logits=True)
        loss = criterion(logits, y_batch)
        
        final_val_loss += loss.item()
        
        _, pred = torch.max(logits, 1)
        final_val_correct += (pred == y_batch).sum().item()
        final_val_total += y_batch.size(0)

final_val_acc = final_val_correct / final_val_total * 100
final_val_loss = final_val_loss / len(val_loader)

print(f"\n最终验证集表现:")
print(f"  损失: {final_val_loss:.4f}")
print(f"  准确率: {final_val_acc:.2f}%")
print(f"  基线准确率: 33.33% (随机猜测)")

if final_val_acc > 40:
    print(f"\n✓ 模型表现超过随机基线 {final_val_acc - 33.33:.2f} 个百分点")
else:
    print(f"\n⚠️  模型表现接近随机，可能仍需改进")

print("\n" + "=" * 90)
print("✓ 训练完成！模型已保存为 model_triclass_alpha.pth")
print("=" * 90)
