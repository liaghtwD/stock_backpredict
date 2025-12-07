"""
三分类模型数据加载器 - 统一版本
集成标签生成 + 数据加载，支持两种模式：
1. 从 features/ 读取预处理好的标签（训练时）
2. 动态生成标签（一次性预处理或即时加载）
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

# ================= 常量 =================
WINDOW_SIZE = 60
PREDICT_DAYS = 20  # 改为20天预测，与策略保持一致

FEATURE_COLS = [
    'zdf', 'hsl', 'volume_change', 'amount_change', 'range', 
    'high_low_ratio', 'close_open_ratio', 'price_MA5_ratio', 'price_MA20_ratio',
    'MACD', 'MACD_signal', 'MACD_hist', 'RSI14', 'BB_width', 'BB_zscore', 'vol_ratio',
    'relative_return', 'beta_60', 'corr_60', 'cum_relative_20', 'cum_relative_60',
    'momentum_60', 'momentum_120', 'volatility_120'
]


# def create_composite_triclass_label(df, predict_days=20):
#     """
#     构建三分类标签 - 基于未来20天的绝对涨跌幅
    
#     标签定义：
#     0: 下跌 - 未来20天内出现-5%的回撤（真正的下跌风险）
#     1: 震荡 - 其他情况
#     2: 上涨 - 未来20天内出现+8%的上涨（真正的上升机会）
    
#     这个定义更简洁，标签分布更均衡
#     """
#     # 计算未来20天的最高价和最低价
#     indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=predict_days)
#     future_high = df['high'].rolling(window=indexer).max()
#     future_low = df['low'].rolling(window=indexer).min()
    
#     current_price = df['close']
    
#     # 计算相对涨幅
#     upside = (future_high - current_price) / current_price
#     downside = (future_low - current_price) / current_price
    
#     # --- 定义下跌 (类别 0) ---
#     # 未来20天内出现 -5% 的回撤
#     is_down = downside < -0.05
    
#     # --- 定义上涨 (类别 2) ---
#     # 未来20天内出现 +8% 的上涨
#     is_up = upside > 0.08
    
#     # --- 整合标签 ---
#     labels = np.ones(len(df), dtype=np.int64)
#     labels[is_down] = 0
#     labels[is_up & (~is_down)] = 2
    
#     return labels


# ================= 数据集类 =================

class StockSequenceDataset(Dataset):
    """三分类数据集"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        # 标签必须是 LongTensor，且范围是 [0, C-1]
        self.labels = torch.LongTensor(labels) 
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
        
    def __len__(self):
        return len(self.labels)

# ================= 数据加载器 (适配新逻辑) =================

def load_stock_data(data_dir, window_size=60, predict_days=20, 
                    split_date=None, feature_cols=None, scaler=None):
    """
    加载数据并使用新的 Alpha 三分类标签
    
    参数:
        scaler: 如果提供，使用该scaler标准化数据（回测时使用）
               如果为None，学习新的scaler（训练时），并返回它
    
    返回:
        X, y, dist, available_cols, scaler_obj
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_features.csv')])
    data_list_x = []
    data_list_y = []
    all_features_for_scaler = []  # 收集所有特征用于拟合scaler
    
    print(f"开始加载数据: 共 {len(csv_files)} 个文件...")
    print(f"使用标签: label_alpha (Alpha + 分位数)")
    
    # ========== 第一遍：如果需要拟合新scaler，先收集所有特征 ==========
    if scaler is None:
        print("第一遍扫描：计算标准化参数...")
        for csv_file in csv_files:
            file_path = os.path.join(data_dir, csv_file)
            try:
                df = pd.read_csv(file_path)
                
                if 'close' not in df.columns: 
                    continue
                
                if split_date:
                    if 'day' in df.columns:
                        df['day'] = pd.to_datetime(df['day'])
                        df = df[df['day'] < pd.Timestamp(split_date)].copy()
                    else:
                        df = df.iloc[:int(len(df) * 0.8)].copy()
                
                if len(df) < window_size + predict_days: 
                    continue
                
                if 'label_alpha' not in df.columns:
                    continue
                
                available_cols = [col for col in feature_cols if col in df.columns]
                if len(available_cols) < 20:
                    continue
                
                feats = df[available_cols].values.astype(np.float32)
                feats = np.nan_to_num(feats, nan=0, posinf=0, neginf=0)
                all_features_for_scaler.append(feats)
                
            except Exception as e:
                continue
        
        if all_features_for_scaler:
            all_features_combined = np.vstack(all_features_for_scaler)
            print(f"  收集到 {len(all_features_combined)} 行特征数据")
            scaler = StandardScaler()
            scaler.fit(all_features_combined)
            print(f"  ✓ 标准化参数已拟合（均值/标准差）")
    
    # ========== 第二遍：用scaler标准化并生成样本 ==========
    print("第二遍扫描：生成训练样本..." if scaler is not None else "生成样本...")
    
    for csv_file in csv_files:
        file_path = os.path.join(data_dir, csv_file)
        try:
            df = pd.read_csv(file_path)
            
            # 基础检查
            if 'close' not in df.columns: 
                continue
            
            # 时间切分
            if split_date:
                if 'day' in df.columns:
                    df['day'] = pd.to_datetime(df['day'])
                    df = df[df['day'] < pd.Timestamp(split_date)].copy()
                else:
                    df = df.iloc[:int(len(df) * 0.8)].copy()
                
            if len(df) < window_size + predict_days: 
                continue
            
            if 'label_alpha' not in df.columns:
                continue
            
            labels = df['label_alpha'].values
            
            # 去除末尾的NaN标签
            valid_idx = ~np.isnan(labels) & (labels >= 0) & (labels <= 2)
            df = df[valid_idx].copy()
            labels = labels[valid_idx]
            
            if len(df) < window_size:
                continue
            
            # 特征检查
            available_cols = [col for col in feature_cols if col in df.columns]
            if len(available_cols) < 20:
                continue
            
            feats = df[available_cols].values.astype(np.float32)
            feats = np.nan_to_num(feats, nan=0, posinf=0, neginf=0)
            
            # 用scaler标准化（使用统一参数）
            if scaler is not None:
                feats = scaler.transform(feats)
            else:
                # 如果没有scaler（仅在测试或调试时发生），用该股票的参数
                temp_scaler = StandardScaler()
                feats = temp_scaler.fit_transform(feats)
            
            # 滑动窗口切片
            for i in range(len(feats) - window_size):
                if i + window_size < len(labels):
                    data_list_x.append(feats[i : i + window_size])
                    data_list_y.append(int(labels[i + window_size - 1]))
                
        except Exception as e:
            continue

    if not data_list_x:
        raise ValueError("未生成数据，请检查路径或标签设置")
        
    X = np.array(data_list_x, dtype=np.float32)
    y = np.array(data_list_y, dtype=np.int64)
    
    # 打印分布
    unique, counts = np.unique(y, return_counts=True)
    dist = dict(zip(unique, counts))
    total = len(y)
    
    print("\n✓ 数据加载完成 (Alpha 三分类标签):")
    print(f"  样本总数: {total}")
    print(f"  [0] 弱势 (Weak):    {dist.get(0, 0):6d} ({dist.get(0, 0)/total*100:5.1f}%)")
    print(f"  [1] 平均 (Average): {dist.get(1, 0):6d} ({dist.get(1, 0)/total*100:5.1f}%)")
    print(f"  [2] 强势 (Strong):  {dist.get(2, 0):6d} ({dist.get(2, 0)/total*100:5.1f}%)")
    
    return X, y, dist, available_cols, scaler


# ================= 计算类别权重 =================
def get_class_weights(y):
    """
    因为震荡样本通常最多，需要计算权重给 Loss 函数，
    让模型更关注稀缺的 上涨 和 下跌 样本
    """
    class_counts = np.bincount(y)
    total_samples = len(y)
    num_classes = len(class_counts)
    
    # sklearn 的 class_weight 计算公式: n_samples / (n_classes * np.bincount(y))
    weights = total_samples / (num_classes * class_counts)
    
    return torch.FloatTensor(weights)