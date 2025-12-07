# model_short/strategy_short.py
import sys
import os

sys.path.append("..")

import pandas as pd
import numpy as np
import joblib


# ================= 工具函数 =================
def calculate_max_drawdown(assets):
    """计算最大回撤"""
    if len(assets) == 0:
        return 0
    peak = max(assets)
    trough = min(assets)
    return (peak - trough) / peak if peak > 0 else 0


def calculate_annual_return(assets, days):
    """计算年化收益率"""
    if len(assets) < 2 or days == 0:
        return 0
    total_return = (assets[-1] - assets[0]) / assets[0]
    annual_return = (1 + total_return) ** (252 / days) - 1
    return annual_return


# ================= 逻辑回归策略 =================
class LogisticShortTermStrategy:
    def __init__(self, model_dir, initial_cash=10000000):
        """
        初始化逻辑回归策略

        Args:
            model_dir: 模型目录，包含每只股票的模型文件
            initial_cash: 初始资金
        """
        self.model_dir = model_dir
        self.initial_cash = initial_cash
        self.loaded_models = {}  # 缓存加载的模型

    def _load_model(self, stock_code):
        """加载指定股票的模型和标准化器"""
        if stock_code in self.loaded_models:
            return self.loaded_models[stock_code]

        try:
            model_path = os.path.join(self.model_dir, f"{stock_code}_model.pkl")
            scaler_path = os.path.join(self.model_dir, f"{stock_code}_scaler.pkl")
            cols_path = os.path.join(self.model_dir, f"{stock_code}_cols.pkl")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")

            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            feature_cols = joblib.load(cols_path)

            self.loaded_models[stock_code] = (model, scaler, feature_cols)
            return model, scaler, feature_cols

        except Exception as e:
            raise Exception(f"加载模型失败 ({stock_code}): {str(e)}")

    def run_backtest(self, stock_code, df_test):
        """对单只股票进行回测

        策略逻辑:
        - 买入: 概率 > 0.55 且系数置信度高时买入
        - 卖出: 概率 < 0.45 或盈利达2%或亏损达1%时卖出
        """
        # 加载模型
        try:
            model, scaler, feature_cols = self._load_model(stock_code)
        except Exception as e:
            print(f"❌ 无法加载 {stock_code} 的模型: {e}")
            return [self.initial_cash] * len(df_test)

        cash = self.initial_cash
        shares = 0
        entry_price = 0
        total_assets = []
        trades = []
        probs = []
        coef_strength = []

        print(f"\n{'=' * 60}")
        print(f"逻辑回归短期策略回测: {stock_code}")
        print(f"{'=' * 60}")
        print(f"时间范围: {df_test['day'].min()} 到 {df_test['day'].max()}")
        print(f"数据行数: {len(df_test)}")
        print(f"使用特征数: {len(feature_cols)}")
        print(f"初始资金: {self.initial_cash:,.0f}")

        # 获取特征数据
        available_cols = [col for col in feature_cols if col in df_test.columns]
        if len(available_cols) < len(feature_cols):
            print(f"⚠️ 警告: 只有 {len(available_cols)}/{len(feature_cols)} 个特征可用")

        # 逐日遍历
        for i in range(len(df_test)):
            current_price = df_test.iloc[i]['close']
            current_day = df_test.iloc[i]['day']

            # 准备特征
            features = []
            for col in feature_cols:
                if col in df_test.columns:
                    features.append(df_test.iloc[i][col])
                else:
                    features.append(0.0)  # 缺失特征用0填充

            features = np.array(features).reshape(1, -1)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            # 标准化并预测
            try:
                features_scaled = scaler.transform(features)
                prob = model.predict_proba(features_scaled)[0][1]

                # 计算系数强度（逻辑回归的优势）
                if hasattr(model, 'coef_'):
                    coef_sum = np.sum(np.abs(model.coef_[0]))
                    if coef_sum > 0:
                        coef_strength.append(coef_sum)
            except:
                prob = 0.5  # 预测失败时使用中性概率

            probs.append(prob)

            # 计算持仓状态
            if shares > 0:
                position_return = (current_price - entry_price) / entry_price
            else:
                position_return = 0

            # 交易决策
            if shares == 0 and prob > 0.55:
                # 买入条件：概率较高且有一定信号强度
                buy_amount = self.initial_cash * 0.5  # 50%仓位
                new_shares = int(buy_amount / current_price)

                if new_shares > 0 and cash >= new_shares * current_price:
                    cash -= new_shares * current_price
                    shares = new_shares
                    entry_price = current_price
                    trades.append(f"[{current_day}] BUY {new_shares} @ {current_price:.2f}, prob={prob:.3f}")

            elif shares > 0:
                sell_reason = None

                # 卖出条件（按优先级）
                if prob < 0.45:
                    sell_reason = f"信号转弱 (prob={prob:.3f})"
                elif position_return > 0.02:  # 盈利2%止盈
                    sell_reason = f"止盈 (盈利{position_return * 100:.1f}%)"
                elif position_return < -0.01:  # 亏损1%止损
                    sell_reason = f"止损 (亏损{abs(position_return) * 100:.1f}%)"

                if sell_reason:
                    cash += shares * current_price
                    profit_pct = position_return * 100
                    trades.append(
                        f"[{current_day}] SELL {shares} @ {current_price:.2f}, {sell_reason}, 盈利={profit_pct:.1f}%")
                    shares = 0
                    entry_price = 0

            # 记录资产净值
            total_assets.append(cash + shares * current_price)

        # 最后结算
        if shares > 0:
            cash += shares * current_price
            total_assets[-1] = cash
            profit_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
            trades.append(f"[最后结算] LIQUIDATE {shares} @ {current_price:.2f}, 盈利={profit_pct:.1f}%")

        # 输出统计
        print(f"\n{'交易记录':-^60}")
        if trades:
            print(f"总共 {len(trades)} 次交易:")
            for trade in trades[:10]:  # 只显示前10条
                print(f"  {trade}")
            if len(trades) > 10:
                print(f"  ... (还有 {len(trades) - 10} 条)")
        else:
            print("没有执行任何交易")

        print(f"\n{'模型表现统计':-^60}")
        if probs:
            prob_array = np.array(probs)
            print(f"平均预测概率: {prob_array.mean():.4f}")
            print(f"概率>0.6的次数: {(prob_array > 0.6).sum()}")
            print(f"概率>0.7的次数: {(prob_array > 0.7).sum()}")

            # 逻辑回归特有的统计
            if coef_strength:
                print(f"平均系数强度: {np.mean(coef_strength):.4f}")
                print(f"强信号天数: {len([p for p in prob_array if p > 0.6])}")

        print(f"\n{'回测结果':-^60}")
        final_assets = total_assets[-1] if total_assets else self.initial_cash
        total_return = (final_assets - self.initial_cash) / self.initial_cash
        max_dd = calculate_max_drawdown(total_assets) if total_assets else 0

        print(f"初始资金:   {self.initial_cash:>15,.0f}")
        print(f"最终资产:   {final_assets:>15,.0f}")
        print(f"总收益:     {total_return:>15.2%}")
        print(f"最大回撤:   {max_dd:>15.2%}")
        print(f"交易次数:   {len(trades):>15}")
        print(f"平均持仓天数: {len(df_test) / max(len(trades) / 2, 1):>15.1f}")
        print(f"{'=' * 60}\n")

        return total_assets


if __name__ == "__main__":
    # 示例：测试一只股票
    stock_code = '000157'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    features_file = os.path.join(script_dir, "..", "features", f"{stock_code}_features.csv")
    model_dir = os.path.join(script_dir, "..", "lr_models")

    if not os.path.exists(features_file):
        print(f"❌ 特征文件不存在: {features_file}")
        print("💡 请先运行 feature.py 生成特征文件")
        exit(1)

    if not os.path.exists(model_dir):
        print(f"❌ 模型目录不存在: {model_dir}")
        print("💡 请先运行 train_short.py 训练模型")
        exit(1)

    df = pd.read_csv(features_file)
    df['day'] = pd.to_datetime(df['day'])
    df_test = df[(df['day'] >= '2024-02-05') & (df['day'] <= '2024-09-20')].sort_values('day')

    if len(df_test) == 0:
        print(f"❌ 没有找到 {stock_code} 在指定期间的数据")
        exit(1)

    strategy = LogisticShortTermStrategy(model_dir)
    assets = strategy.run_backtest(stock_code, df_test)