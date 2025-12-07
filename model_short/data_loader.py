import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ================= 手动指定特征列 =================
# 根据图示，选择有意义的特征列
# 排除用于构建标签的列：high, low, close（用于判断5日内是否触发止盈/止损）
FEATURE_COLS = [
    # --- 基础量价 ---
    'zdf',              # 涨跌幅
    'hsl',              # 换手率
    'volume_change',    # 成交量变化
    'amount_change',    # 成交额变化
    'range',            # 振幅
    'high_low_ratio',   # 高低价比
    'close_open_ratio', # 收开价比
    
    # --- 均线偏离度 (趋势) ---
    'price_MA5_ratio',  # 股价偏离 MA5 程度
    'price_MA20_ratio', # 股价偏离 MA20 程度
    
    # --- 技术指标 (震荡/能量) ---
    'MACD', 
    'MACD_signal', 
    'MACD_hist',
    'RSI14', 
    'BB_width',         # 布林带宽度 (变盘信号)
    'BB_zscore',        # 布林带位置 (超买超卖)
    'vol_ratio',        # 量比 (资金流向)
    
    # --- 相对强弱/Alpha (非常重要) ---
    'relative_return',  # 相对大盘涨跌
    'beta_60',          # 贝塔系数
    'corr_60',          # 相关性
    'cum_relative_20',  # 短期超额动量
    'cum_relative_60',  # 中期超额动量
    
    # --- 动量 (Momentum) ---
    'momentum_60',
    'momentum_120',
    'volatility_120'    # 长期波动率
]

def remove_zero_rows(df, feature_cols):
    """
    舍弃前面包含0值的行，从第一个整行都没有0值的行开始
    
    Args:
        df: 数据框
        feature_cols: 特征列列表
    
    Returns:
        清理后的数据框, 舍弃的行数
    """
    # 检查每行是否包含0值
    has_zero = (df[feature_cols] == 0).any(axis=1)
    no_zero_rows = ~has_zero
    
    if no_zero_rows.any():
        first_no_zero_idx = no_zero_rows.idxmax()
        if first_no_zero_idx > 0:
            df = df.iloc[first_no_zero_idx:].reset_index(drop=True)
            return df, first_no_zero_idx
    
    return df, 0

# ================= 标签构建 =================
def create_short_label(df):
    """
    改进版短期标签 (三重势垒 Triple Barrier):
    利用现有的 high/low 列，寻找未来 5 天内"盈亏比"极佳的机会。
    Target = 1 意味着：在未来5天内，能先摸到止盈线(4%)，且期间不会跌破止损线(-2%)。
    """
    WINDOW = 5
    PROFIT_THR = 1.04  # 止盈 4%
    LOSS_THR = 0.98    # 止损 2% (稍微给一点波动空间)
    
    # 使用 Pandas 的 FixedForwardWindowIndexer 实现"向前"滚动
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=WINDOW)
    
    # 未来 WINDOW 天内的最高价和最低价
    future_max = df['high'].rolling(window=indexer).max()
    future_min = df['low'].rolling(window=indexer).min()
    
    # 1. 获利条件：未来几天最高价冲破了 4%
    cond_profit = future_max > (df['close'] * PROFIT_THR)
    
    # 2. 安全条件：未来几天最低价没有跌破 -2%
    # 这条非常关键！它教会模型避开那些剧烈洗盘或直接暴跌的股票
    cond_safety = future_min > (df['close'] * LOSS_THR)
    
    # === 复合标签 ===
    df['label'] = (cond_profit & cond_safety).astype(int)
    
    return df.dropna(subset=['label'])

# ================= 数据加载 =================
def load_stock_data_short(data_dir, split_date):
    """
    从features目录加载所有CSV文件并构建短期模型训练数据
    
    Args:
        data_dir: features目录路径 (如 '../features')
        split_date: 分割日期 (如 '2023-12-31')，用于训练/测试划分
    
    Returns:
        X, y: 特征和标签数组
        feature_cols: 特征列名列表
    """
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    print(f"检测到 {len(csv_files)} 个CSV文件")
    
    if not csv_files:
        raise ValueError(f"在 {data_dir} 中找不到CSV文件")
    
    data_frames = []
    
    for csv_file in csv_files:
        file_path = os.path.join(data_dir, csv_file)
        
        try:
            df = pd.read_csv(file_path)
            
            # 若没有day列，跳过
            if 'day' not in df.columns:
                continue
            
            # 转换日期
            df['day'] = pd.to_datetime(df['day'])
            
            # 按日期过滤
            if split_date:
                df = df[df['day'] < split_date].copy()
            
            if len(df) < 5:  # 至少要有5条数据用于标签
                continue
            
            # 舍弃包含0值特征的前几行
            df, removed_rows = remove_zero_rows(df, FEATURE_COLS)
            if removed_rows > 0:
                print(f"  {csv_file}: 舍弃前 {removed_rows} 行（包含0值特征）")
            
            if len(df) < 5:  # 清理后要确保仍有足够数据
                continue
            
            # 生成标签
            df = create_short_label(df)
            
            if len(df) == 0:
                continue
            
            data_frames.append(df)
        
        except Exception as e:
            print(f"  警告: 处理 {csv_file} 时出错 - {e}")
            continue
    
    if not data_frames:
        raise ValueError("没有生成任何训练数据，请检查数据文件")
    
    # 合并所有数据
    df_all = pd.concat(data_frames, ignore_index=True)
    
    # 提取特征和标签
    X = df_all[FEATURE_COLS].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    y = df_all['label'].values
    
    print(f"生成样本数: {len(X)}")
    print(f"标签分布: {np.bincount(y.astype(int))}")
    
    return X, y, FEATURE_COLS
