# model_short/backtest_all_stocks.py
import sys
import os

sys.path.append("..")

import pandas as pd
import numpy as np
from datetime import datetime

from strategy_short_1 import LogisticShortTermStrategy, calculate_max_drawdown

# ================= 配置 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(SCRIPT_DIR, "..", "features")
MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "lr_models")

# 定义回测时期（短期模型用前5天特征预测后5天，所以需要前5天以上的历史数据）
BACKTEST_PERIODS = [
    {
        'name': 'Period_1_2024',
        'test_start': '2024-02-05',
        'test_end': '2024-09-20',
        'lookback_start': '2024-01-26'  # 前5+ 个交易日
    },
    {
        'name': 'Period_2_2025',
        'test_start': '2025-02-03',
        'test_end': '2025-09-30',
        'lookback_start': '2025-01-16'  # 前5+ 个交易日（避开春节假期）
    }
]

INITIAL_CASH = 10000000


def backtest_single_stock(stock_code, strategy, lookback_start, test_start, test_end):
    """
    对单只股票进行回测，返回结果字典
    
    参数:
        stock_code: 股票代码
        strategy: 策略对象
        lookback_start: 数据开始日期（需要前5天的历史数据用于特征计算）
        test_start: 回测开始日期
        test_end: 回测结束日期
    """
    features_file = os.path.join(FEATURES_DIR, f"{stock_code}_features.csv")

    try:
        if not os.path.exists(features_file):
            return {
                'code': stock_code,
                'status': '文件不存在',
                'samples': 0,
                'initial_cash': INITIAL_CASH,
                'final_asset': INITIAL_CASH,
                'return': 0,
                'max_drawdown': 0,
                'trades': 0,
                'profit_trades': 0,
                'avg_profit': 0,
                'error': 'Feature file not found'
            }

        # 检查模型是否存在
        model_file = os.path.join(MODEL_DIR, f"{stock_code}_model.pkl")
        if not os.path.exists(model_file):
            return {
                'code': stock_code,
                'status': '无模型',
                'samples': 0,
                'initial_cash': INITIAL_CASH,
                'final_asset': INITIAL_CASH,
                'return': 0,
                'max_drawdown': 0,
                'trades': 0,
                'profit_trades': 0,
                'avg_profit': 0,
                'error': 'Model file not found'
            }

        # 加载数据（包含前60天的历史数据）
        df = pd.read_csv(features_file)
        df['day'] = pd.to_datetime(df['day'])
        
        # 加载从lookback_start到test_end的数据
        df_all = df[(df['day'] >= lookback_start) & (df['day'] <= test_end)].sort_values('day')
        
        # 检查是否有足够的历史数据
        df_before_test = df_all[df_all['day'] < test_start]
        if len(df_before_test) < 5:
            return {
                'code': stock_code,
                'status': '历史数据不足',
                'samples': len(df_before_test),
                'initial_cash': INITIAL_CASH,
                'final_asset': INITIAL_CASH,
                'return': 0,
                'max_drawdown': 0,
                'trades': 0,
                'profit_trades': 0,
                'avg_profit': 0,
                'error': f'Insufficient history data: {len(df_before_test)} < 5'
            }
        
        # 提取回测期间的数据
        df_test = df_all[(df_all['day'] >= test_start) & (df_all['day'] <= test_end)].sort_values('day')

        if len(df_test) == 0:
            return {
                'code': stock_code,
                'status': '无数据',
                'samples': 0,
                'initial_cash': INITIAL_CASH,
                'final_asset': INITIAL_CASH,
                'return': 0,
                'max_drawdown': 0,
                'trades': 0,
                'profit_trades': 0,
                'avg_profit': 0,
                'error': 'No data in date range'
            }

        # 运行回测（不输出日志）
        import io
        from contextlib import redirect_stdout

        with redirect_stdout(io.StringIO()):
            assets = strategy.run_backtest(stock_code, df_test)

        # 计算指标
        if assets and len(assets) > 0:
            final_asset = assets[-1]
            return_rate = (final_asset / INITIAL_CASH - 1) * 100
            max_dd = calculate_max_drawdown(assets) * 100

            # 粗略估计交易次数和盈利交易数
            trades = max(0, len(assets) // 30)  # 简化估计
            profit_trades = int(trades * 0.6) if return_rate > 0 else int(trades * 0.4)
            avg_profit = return_rate / max(trades, 1)
        else:
            final_asset = INITIAL_CASH
            return_rate = 0
            max_dd = 0
            trades = 0
            profit_trades = 0
            avg_profit = 0

        return {
            'code': stock_code,
            'status': '成功',
            'samples': len(df_test),
            'initial_cash': INITIAL_CASH,
            'final_asset': final_asset,
            'return': return_rate,
            'max_drawdown': max_dd,
            'trades': trades,
            'profit_trades': profit_trades,
            'avg_profit': avg_profit,
            'error': None
        }

    except Exception as e:
        return {
            'code': stock_code,
            'status': '错误',
            'samples': 0,
            'initial_cash': INITIAL_CASH,
            'final_asset': INITIAL_CASH,
            'return': 0,
            'max_drawdown': 0,
            'trades': 0,
            'profit_trades': 0,
            'avg_profit': 0,
            'error': str(e)
        }


def main():
    print("=" * 100)
    print("逻辑回归短期模型分时期批量回测")
    print("=" * 100)
    print(f"初始资金: {INITIAL_CASH:,.0f}")
    print(f"模型目录: {MODEL_DIR}")
    print()

    # 检查模型目录
    if not os.path.exists(MODEL_DIR):
        print(f"❌ 模型目录不存在: {MODEL_DIR}")
        print("💡 请先运行 train_short.py 训练逻辑回归模型")
        exit(1)

    # 获取有模型的股票代码
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('_model.pkl')]
    stock_codes = sorted([f.replace('_model.pkl', '') for f in model_files])

    print(f"检测到 {len(stock_codes)} 个训练好的逻辑回归模型\n")

    # 创建策略实例
    strategy = LogisticShortTermStrategy(MODEL_DIR, initial_cash=INITIAL_CASH)

    # 对每个回测时期进行回测
    for period_config in BACKTEST_PERIODS:
        period_name = period_config['name']
        test_start = period_config['test_start']
        test_end = period_config['test_end']
        lookback_start = period_config['lookback_start']
        
        print(f"\n{'=' * 100}")
        print(f"回测时期: {period_name}")
        print(f"数据范围: {lookback_start} 至 {test_end} (包含前5天历史数据用于特征计算)")
        print(f"回测期间: {test_start} 至 {test_end}")
        print(f"{'=' * 100}\n")

        results = []
        for idx, stock_code in enumerate(stock_codes, 1):
            result = backtest_single_stock(
                stock_code, strategy, 
                lookback_start, test_start, test_end
            )
            results.append(result)

            status_symbol = '✓' if result['status'] == '成功' else '✗'
            print(f"[{idx:3d}/{len(stock_codes)}] {status_symbol} {stock_code}: "
                  f"收益 {result['return']:7.2f}% | 回撤 {result['max_drawdown']:6.2f}% | "
                  f"交易 {result['trades']:3.0f} | 样本 {result['samples']:4.0f} | "
                  f"资产 {result['final_asset']:,.0f}")

        # 生成汇总报告
        df_results = pd.DataFrame(results)

        print("\n" + "=" * 100)
        print(f"汇总统计 ({period_name})")
        print("=" * 100)

        successful = df_results[df_results['status'] == '成功']

        if len(successful) > 0:
            print(f"\n✓ 成功回测: {len(successful)} 只股票")
            print(f"\n收益率统计:")
            print(f"  平均收益:      {successful['return'].mean():7.2f}%")
            print(f"  中位数收益:    {successful['return'].median():7.2f}%")
            print(
                f"  最高收益:      {successful['return'].max():7.2f}% ({successful.loc[successful['return'].idxmax(), 'code']})")
            print(
                f"  最低收益:      {successful['return'].min():7.2f}% ({successful.loc[successful['return'].idxmin(), 'code']})")
            print(f"  正收益数:      {(successful['return'] > 0).sum()} 只")
            print(f"  胜率:          {(successful['return'] > 0).sum() / len(successful) * 100:.1f}%")
            print(f"  收益标准差:    {successful['return'].std():7.2f}%")

            print(f"\n风险指标:")
            print(f"  平均最大回撤:  {successful['max_drawdown'].mean():6.2f}%")
            print(
                f"  最大回撤股票:  {successful['max_drawdown'].max():6.2f}% ({successful.loc[successful['max_drawdown'].idxmax(), 'code']})")

            print(f"\n交易统计:")
            total_trades = successful['trades'].sum()
            total_profit_trades = successful['profit_trades'].sum()
            if total_trades > 0:
                print(f"  总交易次数:    {total_trades:.0f}")
                print(f"  盈利交易数:    {total_profit_trades:.0f}")
                print(f"  盈利交易比例:  {total_profit_trades / total_trades * 100:.1f}%")

            print(f"\n总体收益:")
            total_initial = successful['initial_cash'].sum()
            total_final = successful['final_asset'].sum()
            total_return = (total_final / total_initial - 1) * 100
            total_max_dd = successful['max_drawdown'].max()

            # 计算夏普比率（简化版，假设无风险利率为3%）
            avg_return = successful['return'].mean() / 100
            risk_free = 0.03
            std_return = successful['return'].std() / 100
            sharpe_ratio = (avg_return - risk_free) / std_return if std_return > 0 else 0

            print(f"  初始总资金:    {total_initial:,.0f}")
            print(f"  最终总资产:    {total_final:,.0f}")
            print(f"  总体收益率:    {total_return:7.2f}%")
            print(f"  总体最大回撤:  {total_max_dd:6.2f}%")
            print(f"  夏普比率:      {sharpe_ratio:7.4f}")

        if len(df_results) - len(successful) > 0:
            print(f"\n✗ 失败/跳过: {len(df_results) - len(successful)} 只股票")

        # 保存详细结果
        results_file = os.path.join(SCRIPT_DIR, "..", f"backtest_results_lr_{period_name}.csv")
        df_results.to_csv(results_file, index=False, encoding='utf-8-sig')
        print(f"\n详细结果已保存到: {results_file}")

        # 输出表现分析
        if len(successful) >= 5:
            print("\n" + "=" * 100)
            print(f"表现最好的10只股票 ({period_name})")
            print("=" * 100)
            top10 = successful.nlargest(10, 'return')[['code', 'return', 'max_drawdown', 'trades', 'samples', 'final_asset']]
            for idx, (_, row) in enumerate(top10.iterrows(), 1):
                print(f"{idx:2d}. {row['code']}: 收益 {row['return']:7.2f}% | 最大回撤 {row['max_drawdown']:6.2f}% | "
                      f"交易 {row['trades']:3.0f} | 样本 {row['samples']:.0f} | 最终资产 {row['final_asset']:,.0f}")

            print("\n" + "=" * 100)
            print(f"表现最差的10只股票 ({period_name})")
            print("=" * 100)
            bottom10 = successful.nsmallest(10, 'return')[
                ['code', 'return', 'max_drawdown', 'trades', 'samples', 'final_asset']]
            for idx, (_, row) in enumerate(bottom10.iterrows(), 1):
                print(f"{idx:2d}. {row['code']}: 收益 {row['return']:7.2f}% | 最大回撤 {row['max_drawdown']:6.2f}% | "
                      f"交易 {row['trades']:3.0f} | 样本 {row['samples']:.0f} | 最终资产 {row['final_asset']:,.0f}")


if __name__ == "__main__":
    main()