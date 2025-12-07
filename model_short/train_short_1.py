# model_short/train_short.py
import sys
import os

sys.path.append("..")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from model_1 import create_lr_model
from data_loader import load_stock_data_short

# ================= 配置 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "features")
MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "lr_models")  # 为每只股票单独保存模型
SPLIT_DATE = '2023-12-31'  # 训练集截止日期


def train_single_stock(stock_code):
    """为单只股票训练逻辑回归模型"""
    try:
        # 1. 加载该股票的数据
        features_file = os.path.join(DATA_DIR, f"{stock_code}_features.csv")
        if not os.path.exists(features_file):
            print(f"  跳过 {stock_code}: 特征文件不存在")
            return None

        df = pd.read_csv(features_file)
        df['day'] = pd.to_datetime(df['day'])

        # 分割训练集
        df_train = df[df['day'] < SPLIT_DATE].copy()
        if len(df_train) < 50:  # 至少需要50个样本
            print(f"  跳过 {stock_code}: 训练样本不足 ({len(df_train)}个)")
            return None

        # 生成标签
        from data_loader import create_short_label
        df_train = create_short_label(df_train)

        if len(df_train) == 0 or 'label' not in df_train.columns:
            print(f"  跳过 {stock_code}: 无法生成标签")
            return None

        # 准备特征和标签
        feature_cols = [col for col in df_train.columns if col not in ['day', 'label', 'high', 'low', 'close']]

        # 选择数值型特征
        numeric_cols = []
        for col in feature_cols:
            try:
                df_train[col] = pd.to_numeric(df_train[col])
                numeric_cols.append(col)
            except:
                continue

        X = df_train[numeric_cols].values
        y = df_train['label'].values

        # 处理缺失值
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 2. 训练逻辑回归模型
        model = create_lr_model(C=0.5, penalty='l1', max_iter=1000)  # L1正则化进行特征选择
        model.fit(X_scaled, y)

        # 3. 评估
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)[:, 1]

        # 计算指标
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)

        # 4. 保存模型和标准化器
        os.makedirs(MODEL_DIR, exist_ok=True)

        model_path = os.path.join(MODEL_DIR, f"{stock_code}_model.pkl")
        scaler_path = os.path.join(MODEL_DIR, f"{stock_code}_scaler.pkl")
        cols_path = os.path.join(MODEL_DIR, f"{stock_code}_cols.pkl")

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(numeric_cols, cols_path)

        return {
            'stock': stock_code,
            'samples': len(X),
            'pos_rate': (y == 1).sum() / len(y) * 100,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'coef_count': np.sum(np.abs(model.coef_[0]) > 0.001),  # 非零系数数量
            'model_path': model_path
        }

    except Exception as e:
        print(f"  训练 {stock_code} 时出错: {str(e)}")
        return None


def main():
    # 获取所有股票代码
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('_features.csv')]
    stock_codes = sorted([f.replace('_features.csv', '') for f in csv_files])

    print(f"开始为 {len(stock_codes)} 只股票训练逻辑回归模型...")
    print(f"训练集截止日期: {SPLIT_DATE}")
    print(f"模型保存目录: {MODEL_DIR}")
    print("=" * 80)

    results = []
    for idx, stock_code in enumerate(stock_codes, 1):
        print(f"[{idx:3d}/{len(stock_codes)}] 训练 {stock_code}...")
        result = train_single_stock(stock_code)
        if result:
            results.append(result)
            print(f"    样本: {result['samples']:4d}, 正例: {result['pos_rate']:5.1f}%, "
                  f"准确率: {result['accuracy']:.3f}, F1: {result['f1']:.3f}, "
                  f"特征数: {result['coef_count']:2d}")

    # 汇总统计
    if results:
        df_results = pd.DataFrame(results)

        print("\n" + "=" * 80)
        print("训练完成汇总")
        print("=" * 80)
        print(f"成功训练: {len(results)}/{len(stock_codes)} 只股票")
        print(f"平均准确率: {df_results['accuracy'].mean():.4f}")
        print(f"平均F1分数: {df_results['f1'].mean():.4f}")
        print(f"平均正例比例: {df_results['pos_rate'].mean():.2f}%")
        print(f"平均非零特征数: {df_results['coef_count'].mean():.1f}")

        # 保存训练结果
        summary_path = os.path.join(MODEL_DIR, "training_summary.csv")
        df_results.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n详细结果已保存到: {summary_path}")
    else:
        print("没有成功训练任何模型")


if __name__ == "__main__":
    main()