# utils.py
import numpy as np
import pandas as pd
import random
import torch
import os

def setup_seed(seed=42):
    """设置随机种子，保证结果可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def calculate_max_drawdown(wealth_list):
    """计算最大回撤"""
    wealth_array = np.array(wealth_list)
    cumulative_max = np.maximum.accumulate(wealth_array)
    drawdown = (cumulative_max - wealth_array) / cumulative_max
    return np.max(drawdown)

def calculate_annual_return(wealth_list, days):
    """计算年化收益率"""
    if days == 0: return 0
    total_return = wealth_list[-1] / wealth_list[0] - 1
    annual_return = (1 + total_return) ** (252 / days) - 1
    return annual_return

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义模型训练用的特征列 (排除非数值和不适合归一化的列)
# 注意：这里只选比率类、震荡类指标，避免直接使用绝对价格(Open/Close)导致不同股票跨度太大
FEATURE_COLS = [
    'return', 'log_return', 'range', 'high_low_ratio', 'close_open_ratio', 
    'volume_change', 'amount_change', 
    'price_MA5_ratio', 'price_MA20_ratio', 'MACD', 'MACD_signal', 'MACD_hist',
    'RSI14', 'BB_width', 'BB_zscore', 
    'relative_return', 'beta_60', 'corr_60', 'cum_relative_20', 
    'momentum_60', 'momentum_120', 'vol_ratio'
]