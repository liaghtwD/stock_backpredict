"""
预处理三分类标签（改进版）：用 Alpha 标签替换旧标签
基于相对强弱（超额收益）而非绝对涨跌幅，解决原标签过严格导致模型随机的问题
"""
import os
import pandas as pd
import numpy as np
import sys

sys.path.append('.')
from data_loader_triclass import PREDICT_DAYS


def create_alpha_triclass_label(df, market_returns, predict_days=20):
    """
    基于 Alpha（相对强弱）的三分类标签
    
    Alpha = 个股未来收益 - 市场基准未来收益
    按分位数分类：
      - Class 2 (Strong):  Alpha >= 75分位数 (表现最好的前25%)
      - Class 1 (Average): 25分位数 < Alpha < 75分位数 (中间50%)
      - Class 0 (Weak):    Alpha <= 25分位数 (表现最差的前25%)
    
    优点：
    ✓ 适应熊市：即使大家都跌，只要相对跌得少也算Strong
    ✓ 平衡样本：自动维持约 25%-50%-25% 的分布
    ✓ 易于学习：模型学到"相对"特征而非"绝对"特征
    """
    n = len(df)
    labels = np.ones(n, dtype=np.int64)
    
    # 计算个股未来20天的收益
    future_close = df['close'].shift(-predict_days)
    stock_future_return = (future_close - df['close']) / df['close']
    
    # 市场未来收益（累计）
    market_future_return = []
    for i in range(n):
        end_idx = min(i + predict_days, n - 1)
        if end_idx > i:
            # 累计市场收益
            mkt_ret = (1 + market_returns[i:end_idx]).prod() - 1
        else:
            mkt_ret = 0
        market_future_return.append(mkt_ret)
    
    market_future_return = np.array(market_future_return)
    
    # 计算Alpha
    alpha = stock_future_return - market_future_return
    
    # 过滤掉NaN和Inf
    valid_mask = ~np.isnan(alpha) & ~np.isinf(alpha)
    valid_alpha = alpha[valid_mask]
    
    if len(valid_alpha) == 0:
        return labels
    
    # 计算分位数
    alpha_25 = np.percentile(valid_alpha, 25)
    alpha_75 = np.percentile(valid_alpha, 75)
    
    # 分配标签
    labels[alpha >= alpha_75] = 2  # Strong (前25%)
    labels[alpha <= alpha_25] = 0  # Weak (后25%)
    # labels[中间] = 1  # Average (已初始化)
    
    return labels


def preprocess_triclass_labels(features_dir='../features', market_returns=None, force_regenerate=True):
    """
    为所有特征文件添加改进的 Alpha 三分类标签列
    
    Args:
        features_dir: 特征文件目录
        market_returns: 市场收益率数组（按日期顺序）
        force_regenerate: 如果为True，强制重新生成所有标签
    """
    print("="*80)
    print("预处理三分类标签（Alpha版本）")
    print("="*80)
    print(f"预测天数: {PREDICT_DAYS} 天")
    print(f"标签方式: 基于相对强弱（Alpha）+ 分位数分类")
    
    # 如果未提供市场收益，从第一个特征文件中获取
    if market_returns is None:
        print("\n正在获取市场基准数据...")
        stock_files = [f for f in os.listdir(features_dir) if f.endswith('_features.csv')]
        first_file = os.path.join(features_dir, stock_files[0])
        df_first = pd.read_csv(first_file)
        market_returns = df_first['market_return'].values
        print(f"✓ 从 {stock_files[0]} 获取市场基准 ({len(market_returns)} 条记录)")
    
    if not os.path.exists(features_dir):
        print(f"❌ 目录不存在: {features_dir}")
        return
    
    csv_files = sorted([f for f in os.listdir(features_dir) if f.endswith('_features.csv')])
    print(f"✓ 发现 {len(csv_files)} 个特征文件\n")
    
    success_count = 0
    fail_count = 0
    label_stats = []
    
    for idx, csv_file in enumerate(csv_files, 1):
        file_path = os.path.join(features_dir, csv_file)
        stock_code = csv_file.replace('_features.csv', '')
        
        try:
            # 读取特征文件
            df = pd.read_csv(file_path)
            
            # 检查必要的列
            if 'close' not in df.columns:
                fail_count += 1
                continue
            
            # 删除旧标签列
            if 'label_alpha' in df.columns:
                df = df.drop('label_alpha', axis=1)
            if 'triclass_label' in df.columns:
                df = df.drop('triclass_label', axis=1)
            
            # 创建 Alpha 三分类标签
            labels = create_alpha_triclass_label(df, market_returns, predict_days=PREDICT_DAYS)
            
            # 添加标签列
            df['label_alpha'] = labels
            
            # 保存文件
            df.to_csv(file_path, index=False)
            
            # 统计标签分布
            label_counts = pd.Series(labels).value_counts().sort_index()
            weak_count = label_counts.get(0, 0)
            avg_count = label_counts.get(1, 0)
            strong_count = label_counts.get(2, 0)
            total = len(labels)
            
            label_stats.append({
                'stock': stock_code,
                'weak_0': weak_count,
                'avg_1': avg_count,
                'strong_2': strong_count,
                'weak_%': weak_count / total * 100,
                'avg_%': avg_count / total * 100,
                'strong_%': strong_count / total * 100,
            })
            
            success_count += 1
            
            if idx % 10 == 0 or idx == len(csv_files):
                print(f"  ✓ {idx:2d}/{len(csv_files)} {stock_code}: "
                      f"弱={weak_count:4d} ({weak_count/total*100:5.1f}%) | "
                      f"平={avg_count:4d} ({avg_count/total*100:5.1f}%) | "
                      f"强={strong_count:4d} ({strong_count/total*100:5.1f}%)")
        
        except Exception as e:
            fail_count += 1
            if idx <= 5 or idx > len(csv_files) - 5:  # 只打印开头和结尾的错误
                print(f"  ❌ {idx:2d}/{len(csv_files)} {stock_code}: {str(e)[:60]}")
            continue
    
    # 汇总统计
    print(f"\n" + "="*80)
    print(f"处理结果: 成功 {success_count}/{len(csv_files)}，失败 {fail_count}/{len(csv_files)}")
    print("="*80)
    
    if label_stats:
        stats_df = pd.DataFrame(label_stats)
        print(f"\n标签分布汇总统计：")
        print(f"  弱势类（Class 0）平均占比: {stats_df['weak_%'].mean():.1f}% "
              f"(范围: {stats_df['weak_%'].min():.1f}% ~ {stats_df['weak_%'].max():.1f}%)")
        print(f"  平均类（Class 1）平均占比: {stats_df['avg_%'].mean():.1f}% "
              f"(范围: {stats_df['avg_%'].min():.1f}% ~ {stats_df['avg_%'].max():.1f}%)")
        print(f"  强势类（Class 2）平均占比: {stats_df['strong_%'].mean():.1f}% "
              f"(范围: {stats_df['strong_%'].min():.1f}% ~ {stats_df['strong_%'].max():.1f}%)")
        
        # 保存统计
        stats_df.to_csv('label_distribution_alpha.csv', index=False)
        print(f"\n✓ 标签分布统计已保存到 label_distribution_alpha.csv")


if __name__ == '__main__':
    # 确定正确的特征目录路径
    features_dir = '../features'
    
    if not os.path.exists(features_dir):
        print(f"❌ 找不到特征目录: {features_dir}")
        exit(1)
    
    print(f"✓ 特征目录: {os.path.abspath(features_dir)}\n")
    
    # 预处理标签
    preprocess_triclass_labels(features_dir=features_dir)
