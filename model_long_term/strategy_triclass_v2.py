"""CLI-friendly wrapper around the shared triclass strategy implementation."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.append('.')

from strategies.triclass_core import TriclassStrategy


class TriclassLongTermStrategy:
    """提供与旧 CLI 一致的界面，同时复用核心策略实现。"""

    def __init__(
        self,
        model_path: str = 'model_triclass_alpha.pth',
        scaler_path: str = 'scaler_triclass.pkl',
        initial_capital: float = 10_000_000,
        thresholds: dict | None = None,
    ) -> None:
        self.initial_capital = initial_capital

        base_thresholds = thresholds or {
            'entry_up_threshold': 0.55,
            'entry_down_cap': 0.30,
            'entry_margin': 0.18,
            'add_up_threshold': 0.64,
            'exit_down_threshold': 0.53,
        }

        position_config = {
            'initial_buy_ratio': 0.5,
            'add_buy_ratio': 0.3,
            'max_trades_total': 20,
            'max_buys_in_window': 5,
            'recent_buy_lookback_days': 10,
            'hard_stop_loss': 0.08,
            'trailing_min_profit': 0.08,
            'trailing_drawdown': 0.08,
            'time_stop_days': 10,
            'time_stop_band': 0.02,
            'partial_take_profit_ratio': 0.3,
        }

        cost_config = {'commission': 0.0, 'tax': 0.0, 'slippage': 0.0}

        self.strategy = TriclassStrategy(
            model_path=model_path,
            scaler_path=scaler_path,
            default_config=base_thresholds,
            position_config=position_config,
            cost_config=cost_config,
        )

        print("✓ 三分类策略已初始化 (复用核心模块)")
        print(f"  模型路径: {os.path.abspath(model_path)}")
        print(f"  初始资金: {initial_capital:,.0f}")
        print(f"  默认阈值: {base_thresholds}")

    def backtest_single_stock(self, df: pd.DataFrame, stock_code: str) -> dict:
        """
        对单只股票进行独立回测
        
        Args:
            df: DataFrame，包含所有需要的列（含特征列）
            stock_code: 股票代码
        
        Returns:
            {
                'stock': 股票代码,
                'total_return': 总收益率(%),
                'annual_return': 年化收益率(%),
                'max_drawdown': 最大回撤(%),
                'num_trades': 交易次数,
                'win_rate': 胜率(%),
                'avg_profit': 平均单笔利润(%),
                'trades': 交易详情列表,
                'equity_curve': 净值曲线,
                'asset_curve': 资产曲线
            }
        """
        print(f"\n{'='*70}")
        print(f"三分类策略回测: {stock_code}")
        print(f"{'='*70}")
        print(f"数据范围: {df.index[0].strftime('%Y-%m-%d')} 至 {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"数据行数: {len(df)}")
        print(f"初始资金: {self.initial_capital:,.0f}")

        result = self.strategy.backtest_stock(
            df,
            stock_code,
            initial_capital=self.initial_capital,
            include_details=True,
        )

        if result.get('error'):
            print("❌ 回测失败，可能是数据不足或特征缺失")
            return {
                'stock': stock_code,
                'total_return': 0,
                'annual_return': 0,
                'max_drawdown': 0,
                'num_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'trades': [],
                'equity_curve': np.array([]),
                'asset_curve': np.array([]),
                'final_assets': self.initial_capital,
                'error': result.get('error', '数据不足或特征缺失'),
            }

        trades = result.get('trades', [])
        asset_curve = np.array(result.get('asset_curve', []), dtype=float)
        equity_curve = asset_curve.copy()

        sell_trades = [t for t in trades if t['type'].startswith('SELL')]
        winning_trades = [t for t in sell_trades if t.get('profit', 0) > 0]
        # 核心模块以收益百分比形式返回 profit 字段（建仓为0），保持兼容
        win_rate = len(winning_trades) / len(sell_trades) * 100 if sell_trades else 0
        avg_profit = np.mean([t.get('profit', 0) for t in sell_trades]) if sell_trades else 0

        print(f"\n{'交易记录':-^70}")
        print(f"总交易数: {len(trades)}")
        if trades:
            for idx, trade in enumerate(trades[:15]):
                date_obj = trade.get('date')
                date_str = date_obj.strftime('%Y-%m-%d') if hasattr(date_obj, 'strftime') else str(date_obj)
                profit_val = trade.get('profit', 0)
                print(
                    f"  [{idx+1:2d}] {date_str} {trade['type']:<18} "
                    f"@{trade['price']:>7.2f} x{trade['shares']:>6d} "
                    f"利润={profit_val:>+6.2f}%"
                )
            if len(trades) > 15:
                print(f"  ... 还有 {len(trades) - 15} 笔交易")

        print(f"\n{'回测结果统计':-^70}")
        print(f"初始资金:    {self.initial_capital:>15,.0f} 元")
        print(f"最终资产:    {result['final_asset']:>15,.0f} 元")
        print(f"总收益:      {result['total_return']:>15.2f}%")
        print(f"年化收益:    {result['annual_return']:>15.2f}%")
        print(f"最大回撤:    {result['max_drawdown']:>15.2f}%")
        print(f"交易次数:    {result['num_trades']:>15}")
        print(f"胜率:        {win_rate:>15.1f}%")
        print(f"平均利润:    {avg_profit:>15.2f}%")
        print(f"{'='*70}\n")

        return {
            'stock': stock_code,
            'total_return': result['total_return'],
            'annual_return': result['annual_return'],
            'max_drawdown': result['max_drawdown'],
            'num_trades': result['num_trades'],
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'trades': trades,
            'equity_curve': equity_curve,
            'asset_curve': asset_curve,
            'final_assets': result['final_asset'],
        }
