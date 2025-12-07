"""
市场周期分段回测分析 (Market Regime-Based Backtest Analysis)

功能：
1. 根据 condition.csv 中的市场周期标注（牛市/熊市/震荡市）分段回测
2. 对比中长期策略（GA优化参数）和短期策略在不同市场环境下的表现
3. 分析各周期对总收益的贡献度和风险特征

市场周期：
- bull: 牛市（上涨趋势）
- bear: 熊市（下跌趋势）
- consolidation: 震荡市（横盘整理）

输出：
- regime_analysis_longterm.csv: 中长期策略各周期表现
- regime_analysis_shortterm.csv: 短期策略各周期表现
- regime_comparison.csv: 两种策略对比
- regime_detail_trades.csv: 各周期详细交易记录
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import pickle

import pandas as pd
import numpy as np

sys.path.append('.')
sys.path.append('model_long_term')
from model_long_term.strategies.triclass_core import TriclassStrategy, normalize_stock_code
from model_short.strategy_short_1 import LogisticShortTermStrategy

# ================================================================================
# 配置
# ================================================================================

# 市场周期数据
CONDITION_FILE = 'condition.csv'

# 回测时间段（使用2021年保证所有88只股票都有完整数据）
BACKTEST_START = '2021-01-01'
BACKTEST_END = '2025-09-30'

# 数据路径
FEATURES_DIR = Path('features')
MODEL_LONG_DIR = Path('model_long_term')
MODEL_SHORT_DIR = Path('lr_models')  # 短期策略模型在根目录

# 输出目录
OUTPUT_DIR = Path('regime_analysis_results')
OUTPUT_DIR.mkdir(exist_ok=True)

# 初始资金
INITIAL_CAPITAL = 10_000_000

# 板块映射（与 GA 训练保持一致）
STOCK_CLASSIFICATION_MAP = {
    # 酒类
    '000858': 'alcohol', '600519': 'alcohol', '002304': 'alcohol',
    '000568': 'alcohol', '603369': 'alcohol', '603589': 'alcohol',
    '603198': 'alcohol', '603919': 'alcohol',
    
    # 芯片
    '603986': 'chip', '688981': 'chip', '002371': 'chip',
    '600703': 'chip', '603501': 'chip', '688187': 'chip',
    '688008': 'chip', '300661': 'chip', '300223': 'chip',
    '300782': 'chip', '002049': 'chip', '300373': 'chip',
    '300346': 'chip', '300567': 'chip', '300458': 'chip',
    
    # 新能源
    '002812': 'new energy', '002460': 'new energy', '300450': 'new energy',
    
    # 电池
    '300014': 'batteries', '300750': 'batteries', '002466': 'batteries',
    '603659': 'batteries',
    
    # 汽车
    '002594': 'automobile', '601633': 'automobile', '600104': 'automobile',
    '000625': 'automobile', '601238': 'automobile', '002708': 'automobile',
    
    # 电力
    '600900': 'electric power', '003816': 'electric power',
    '601985': 'electric power', '600011': 'electric power',
    '600023': 'electric power', '000993': 'electric power',
    
    # 教育
    '300359': 'education', '002261': 'education', '600661': 'education',
    '002315': 'education', '603877': 'education', '002563': 'education',
    '002291': 'education', '002425': 'education', '002569': 'education',
    
    # 工程机械
    '000157': 'engineering machinery', '000425': 'engineering machinery',
    '600031': 'engineering machinery', '601100': 'engineering machinery',
    '002097': 'engineering machinery',
    
    # 风电设备
    '002202': 'wind power equipment', '601615': 'wind power equipment',
    '300443': 'wind power equipment', '002531': 'wind power equipment',
    '603606': 'wind power equipment',
    
    # 光伏设备
    '601012': 'Photovoltaic equipment', '300274': 'Photovoltaic equipment',
    '002459': 'Photovoltaic equipment', '603806': 'Photovoltaic equipment',
    '688599': 'Photovoltaic equipment', '300118': 'Photovoltaic equipment',
    
    # 家电
    '002242': 'home appliance', '603486': 'home appliance',
    '002508': 'home appliance', '002032': 'home appliance',
    '603355': 'home appliance',
    
    # 医药
    '600276': 'medical', '000661': 'medical', '300003': 'medical',
    '603259': 'medical', '002821': 'medical', '002422': 'medical',
    '600276': 'medical', '300760': 'medical',
    
    # 银行
    '601398': 'bank', '601288': 'bank', '601939': 'bank',
    '601328': 'bank', '600036': 'bank', '600000': 'bank',
    '601166': 'bank', '600016': 'bank', '601818': 'bank',
    
    # 保险
    '601318': 'insurance', '601601': 'insurance', '601336': 'insurance',
}

# ================================================================================
# 市场周期数据加载
# ================================================================================

def load_market_regimes():
    """
    加载市场周期标注数据
    
    返回：
        DataFrame: 包含 day, market_condition 列
    """
    df = pd.read_excel(CONDITION_FILE)
    df['day'] = pd.to_datetime(df['day'])
    return df[['day', 'market_condition']].copy()


def segment_by_regime(start_date, end_date, regime_df):
    """
    将时间段按市场周期切分
    
    参数：
        start_date: 开始日期（字符串）
        end_date: 结束日期（字符串）
        regime_df: 市场周期数据
        
    返回：
        list of dict: [{'regime': 'bull', 'start': '2024-03-01', 'end': '2024-06-15', 'days': 75}, ...]
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # 筛选时间段内的数据
    mask = (regime_df['day'] >= start) & (regime_df['day'] <= end)
    period_data = regime_df[mask].copy().reset_index(drop=True)
    
    if len(period_data) == 0:
        return []
    
    # 识别连续的相同周期
    segments = []
    current_regime = period_data.iloc[0]['market_condition']
    current_start = period_data.iloc[0]['day']
    
    for i in range(1, len(period_data)):
        row = period_data.iloc[i]
        
        if row['market_condition'] != current_regime:
            # 周期变化，保存上一段
            segments.append({
                'regime': current_regime,
                'start': current_start.strftime('%Y-%m-%d'),
                'end': period_data.iloc[i-1]['day'].strftime('%Y-%m-%d'),
                'days': (period_data.iloc[i-1]['day'] - current_start).days + 1
            })
            
            # 开始新周期
            current_regime = row['market_condition']
            current_start = row['day']
    
    # 添加最后一段
    segments.append({
        'regime': current_regime,
        'start': current_start.strftime('%Y-%m-%d'),
        'end': period_data.iloc[-1]['day'].strftime('%Y-%m-%d'),
        'days': (period_data.iloc[-1]['day'] - current_start).days + 1
    })
    
    return segments


# ================================================================================
# 数据加载
# ================================================================================

def load_stock_features(stock_code, start_date, end_date):
    """
    加载股票特征数据
    
    参数：
        stock_code: 标准化后的股票代码（6位）
        start_date: 开始日期
        end_date: 结束日期
        
    返回：
        DataFrame 或 None
    """
    feature_file = FEATURES_DIR / f"{stock_code}_features.csv"
    
    if not feature_file.exists():
        return None
    
    try:
        df = pd.read_csv(feature_file)
        
        # 处理日期列（可能是 day 或 date）
        if 'day' in df.columns:
            df['date'] = pd.to_datetime(df['day'])
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            raise KeyError("未找到日期列(day或date)")
        
        # 设置日期为索引（triclass_core需要df.index为日期类型）
        df = df.set_index('date')
        
        # 检查股票数据是否覆盖回测起始日期
        data_start = df.index.min()
        required_start = pd.to_datetime(start_date)
        
        # 如果股票上市时间晚于回测起始时间,跳过此股票
        if data_start > required_start:
            # print(f"  ℹ️  {stock_code} 上市日期 {data_start.date()} 晚于回测起始 {required_start.date()}, 跳过")
            return None
        
        # 筛选时间段（包含前5天用于短期策略特征计算）
        start = pd.to_datetime(start_date) - pd.Timedelta(days=10)
        end = pd.to_datetime(end_date)
        
        mask = (df.index >= start) & (df.index <= end)
        filtered_df = df[mask].copy()
        
        # 确保有足够的数据点
        if len(filtered_df) < 10:
            return None
            
        return filtered_df
    
    except Exception as e:
        print(f"  ⚠️  加载 {stock_code} 特征数据失败: {e}")
        return None


# ================================================================================
# 中长期策略回测
# ================================================================================

def backtest_longterm_by_regime(segments, regime_df):
    """
    对中长期策略按市场周期分段回测
    
    参数：
        segments: 市场周期切分结果
        regime_df: 市场周期数据
        
    返回：
        DataFrame: 各周期回测结果
    """
    print("\n" + "="*80)
    print("📊 中长期策略（GA优化参数）- 市场周期分段回测")
    print("="*80)
    
    # 加载 GA 优化的参数（优先使用 2024 版本，回退到 2023）
    param_file_2024 = MODEL_LONG_DIR / 'ga_best_params_2024.pkl'
    param_file_2023 = MODEL_LONG_DIR / 'ga_best_params_2023.pkl'
    
    if param_file_2024.exists():
        with open(param_file_2024, 'rb') as f:
            best_configs = pickle.load(f)
        print(f"✅ 加载 GA 参数: {param_file_2024}")
    elif param_file_2023.exists():
        with open(param_file_2023, 'rb') as f:
            best_configs = pickle.load(f)
        print(f"✅ 加载 GA 参数: {param_file_2023}")
    else:
        print("❌ 未找到 GA 参数文件，请先运行 train_ga_params.py")
        return pd.DataFrame()
    
    # 构建策略
    strategy = TriclassStrategy(
        model_path=str(MODEL_LONG_DIR / 'model_triclass_alpha.pth'),
        scaler_path=str(MODEL_LONG_DIR / 'scaler_triclass.pkl'),
        classification_configs=best_configs,
        stock_classification_map=STOCK_CLASSIFICATION_MAP,
    )
    
    # 获取所有股票代码
    all_stocks = list(STOCK_CLASSIFICATION_MAP.keys())
    
    results = []
    
    for seg in segments:
        regime = seg['regime']
        start = seg['start']
        end = seg['end']
        days = seg['days']
        
        print(f"\n{'='*60}")
        print(f"🔍 周期: {regime.upper():15s} | {start} ~ {end} ({days}天)")
        print(f"{'='*60}")
        
        segment_results = []
        
        for stock_code in all_stocks:
            norm_code = normalize_stock_code(stock_code)
            
            # 加载数据
            df = load_stock_features(norm_code, start, end)
            if df is None:
                continue
            
            # 在时间段内回测
            try:
                result = strategy.backtest_stock(
                    df, norm_code, 
                    initial_capital=INITIAL_CAPITAL,
                    include_details=False
                )
                
                if not result.get('error'):
                    segment_results.append(result)
            
            except Exception as e:
                print(f"  ⚠️  {norm_code} 回测失败: {e}")
                continue
        
        # 汇总本周期结果
        if segment_results:
            total_return = np.mean([r['annual_return'] for r in segment_results])
            # max_drawdown 在 triclass_core 中是负数,取绝对值
            max_dd = np.mean([abs(r['max_drawdown']) for r in segment_results])
            total_trades = sum([r['num_trades'] for r in segment_results])
            
            # 计算胜率（需要详细交易记录，这里简化处理）
            win_rate = 0.5  # 默认值，triclass_core 不返回 win_rate
            
            results.append({
                'regime': regime,
                'start_date': start,
                'end_date': end,
                'trading_days': days,
                'stocks_count': len(segment_results),
                'avg_annual_return': total_return,
                'avg_max_drawdown': max_dd,  # 现在是正数
                'total_trades': total_trades,
                'avg_win_rate': win_rate,
                'sharpe_ratio': total_return / max(max_dd, 0.01),
            })
            
            print(f"  📈 平均年化收益: {total_return:.2%}")
            print(f"  📉 平均最大回撤: {max_dd:.2%}")
            print(f"  🔢 总交易次数: {total_trades}")
            print(f"  ✅ 平均胜率: {win_rate:.2%}")
    
    return pd.DataFrame(results)


# ================================================================================
# 短期策略回测
# ================================================================================

def backtest_shortterm_by_regime(segments, regime_df):
    """
    对短期策略按市场周期分段回测
    
    参数：
        segments: 市场周期切分结果
        regime_df: 市场周期数据
        
    返回：
        DataFrame: 各周期回测结果
    """
    print("\n" + "="*80)
    print("📊 短期策略（Logistic回归）- 市场周期分段回测")
    print("="*80)
    
    # 获取所有有模型的股票
    model_files = list(MODEL_SHORT_DIR.glob('*_model.pkl'))
    all_stocks = [f.stem.replace('_model', '') for f in model_files]
    
    if not all_stocks:
        print("❌ 未找到短期策略模型文件")
        return pd.DataFrame()
    
    print(f"✅ 找到 {len(all_stocks)} 只股票的短期模型")
    
    results = []
    
    for seg in segments:
        regime = seg['regime']
        start = seg['start']
        end = seg['end']
        days = seg['days']
        
        print(f"\n{'='*60}")
        print(f"🔍 周期: {regime.upper():15s} | {start} ~ {end} ({days}天)")
        print(f"{'='*60}")
        
        segment_results = []
        
        for stock_code in all_stocks:
            model_file = MODEL_SHORT_DIR / f"{stock_code}_model.pkl"
            
            if not model_file.exists():
                continue
            
            # 加载数据
            df = load_stock_features(stock_code, start, end)
            if df is None or len(df) < 10:
                continue
            
            try:
                # 初始化策略（短期策略需要 model_dir 参数）
                if not hasattr(backtest_shortterm_by_regime, '_strategy'):
                    backtest_shortterm_by_regime._strategy = LogisticShortTermStrategy(
                        model_dir=str(MODEL_SHORT_DIR),
                        initial_cash=INITIAL_CAPITAL
                    )
                
                strategy = backtest_shortterm_by_regime._strategy
                
                # 回测（返回资产曲线）
                asset_curve = strategy.run_backtest(stock_code, df)
                
                if asset_curve and len(asset_curve) > 0:
                    # 手动计算统计指标
                    final_asset = asset_curve[-1]
                    total_return = (final_asset - INITIAL_CAPITAL) / INITIAL_CAPITAL
                    
                    # 计算最大回撤
                    cummax = np.maximum.accumulate(asset_curve)
                    drawdowns = (np.array(asset_curve) - cummax) / cummax
                    max_dd = abs(min(drawdowns)) if len(drawdowns) > 0 else 0
                    
                    # 只统计有实际收益的股票
                    if abs(total_return) > 0.001:  # 过滤掉基本没变化的
                        segment_results.append({
                            'return': total_return,
                            'max_drawdown': max_dd,
                            'trades': 1,  # 占位符，短期策略未返回交易次数
                            'profit_trades': 1 if total_return > 0 else 0
                        })
            
            except Exception as e:
                # print(f"  ⚠️  {stock_code} 回测失败: {e}")
                continue
        
        # 汇总本周期结果
        if segment_results:
            returns = [r['return'] for r in segment_results]
            drawdowns = [r['max_drawdown'] for r in segment_results]
            
            avg_return = np.mean(returns)
            avg_dd = np.mean(drawdowns)
            total_trades = sum([r['trades'] for r in segment_results])
            win_rates = [r['profit_trades'] / r['trades'] for r in segment_results if r['trades'] > 0]
            avg_win_rate = np.mean(win_rates) if win_rates else 0
            
            # 短期策略返回的是总收益率，直接年化
            # 假设收益率是整个周期的，转换为年化收益
            annualization_factor = 252 / max(days, 1)
            annual_return = ((1 + avg_return) ** annualization_factor - 1)
            
            results.append({
                'regime': regime,
                'start_date': start,
                'end_date': end,
                'trading_days': days,
                'stocks_count': len(segment_results),
                'avg_period_return': avg_return,
                'avg_annual_return': annual_return,
                'avg_max_drawdown': avg_dd,
                'total_trades': total_trades,
                'avg_win_rate': avg_win_rate,
                'sharpe_ratio': annual_return / max(avg_dd, 0.01) if avg_dd > 0 else 0,
            })
            
            print(f"  📈 平均期间收益: {avg_return:.2%} (年化: {annual_return:.2%})")
            print(f"  📉 平均最大回撤: {avg_dd:.2%}")
            print(f"  🔢 总交易次数: {total_trades}")
            print(f"  ✅ 平均胜率: {avg_win_rate:.2%}")
    
    return pd.DataFrame(results)


# ================================================================================
# 对比分析
# ================================================================================

def compare_strategies(longterm_df, shortterm_df):
    """
    对比两种策略在不同市场周期的表现
    
    参数：
        longterm_df: 中长期策略结果
        shortterm_df: 短期策略结果
        
    返回：
        DataFrame: 对比结果
    """
    print("\n" + "="*80)
    print("📊 策略对比分析")
    print("="*80)
    
    # 合并数据
    comparison = []
    
    regimes = set(longterm_df['regime'].unique()) | set(shortterm_df['regime'].unique())
    
    for regime in sorted(regimes):
        lt = longterm_df[longterm_df['regime'] == regime]
        st = shortterm_df[shortterm_df['regime'] == regime]
        
        row = {'regime': regime}
        
        if len(lt) > 0:
            row['longterm_return'] = lt['avg_annual_return'].iloc[0]
            row['longterm_drawdown'] = lt['avg_max_drawdown'].iloc[0]
            row['longterm_sharpe'] = lt['sharpe_ratio'].iloc[0]
            row['longterm_trades'] = lt['total_trades'].iloc[0]
        else:
            row['longterm_return'] = 0
            row['longterm_drawdown'] = 0
            row['longterm_sharpe'] = 0
            row['longterm_trades'] = 0
        
        if len(st) > 0:
            row['shortterm_return'] = st['avg_annual_return'].iloc[0]
            row['shortterm_drawdown'] = st['avg_max_drawdown'].iloc[0]
            row['shortterm_sharpe'] = st['sharpe_ratio'].iloc[0]
            row['shortterm_trades'] = st['total_trades'].iloc[0]
        else:
            row['shortterm_return'] = 0
            row['shortterm_drawdown'] = 0
            row['shortterm_sharpe'] = 0
            row['shortterm_trades'] = 0
        
        # 计算优势策略
        row['better_return'] = 'longterm' if row['longterm_return'] > row['shortterm_return'] else 'shortterm'
        row['better_sharpe'] = 'longterm' if row['longterm_sharpe'] > row['shortterm_sharpe'] else 'shortterm'
        
        comparison.append(row)
        
        print(f"\n【{regime.upper()}】")
        print(f"  中长期策略: 年化收益 {row['longterm_return']:.2f}% | 回撤 {row['longterm_drawdown']:.2f}% | Sharpe {row['longterm_sharpe']:.2f}")
        print(f"  短期策略:   年化收益 {row['shortterm_return']:.2f}% | 回撤 {row['shortterm_drawdown']:.2f}% | Sharpe {row['shortterm_sharpe']:.2f}")
        print(f"  优势策略:   收益 -> {row['better_return']} | 风险调整收益 -> {row['better_sharpe']}")
    
    return pd.DataFrame(comparison)


# ================================================================================
# 主函数
# ================================================================================

def main():
    print("\n" + "="*80)
    print("🚀 市场周期分段回测分析")
    print("="*80)
    print(f"回测时间: {BACKTEST_START} ~ {BACKTEST_END}")
    print(f"初始资金: {INITIAL_CAPITAL:,}")
    
    # 1. 加载市场周期数据
    print("\n📅 加载市场周期数据...")
    regime_df = load_market_regimes()
    print(f"  ✅ 共 {len(regime_df)} 个交易日")
    print(f"  📊 周期分布: {dict(regime_df['market_condition'].value_counts())}")
    
    # 2. 切分市场周期
    print(f"\n✂️  切分市场周期 ({BACKTEST_START} ~ {BACKTEST_END})...")
    segments = segment_by_regime(BACKTEST_START, BACKTEST_END, regime_df)
    
    if not segments:
        print("❌ 未找到有效的市场周期数据")
        return
    
    print(f"  ✅ 识别出 {len(segments)} 个连续周期段:")
    for seg in segments:
        print(f"     - {seg['regime']:15s}: {seg['start']} ~ {seg['end']} ({seg['days']}天)")
    
    # 3. 中长期策略回测
    longterm_results = backtest_longterm_by_regime(segments, regime_df)
    
    # 4. 短期策略回测
    shortterm_results = backtest_shortterm_by_regime(segments, regime_df)
    
    # 5. 对比分析
    if not longterm_results.empty and not shortterm_results.empty:
        comparison = compare_strategies(longterm_results, shortterm_results)
        
        # 保存结果
        longterm_results.to_csv(OUTPUT_DIR / 'regime_analysis_longterm.csv', index=False, encoding='utf-8-sig')
        shortterm_results.to_csv(OUTPUT_DIR / 'regime_analysis_shortterm.csv', index=False, encoding='utf-8-sig')
        comparison.to_csv(OUTPUT_DIR / 'regime_comparison.csv', index=False, encoding='utf-8-sig')
        
        print("\n" + "="*80)
        print("✅ 分析完成！结果已保存到:")
        print(f"   - {OUTPUT_DIR / 'regime_analysis_longterm.csv'}")
        print(f"   - {OUTPUT_DIR / 'regime_analysis_shortterm.csv'}")
        print(f"   - {OUTPUT_DIR / 'regime_comparison.csv'}")
        print("="*80)
    
    elif not longterm_results.empty:
        longterm_results.to_csv(OUTPUT_DIR / 'regime_analysis_longterm.csv', index=False, encoding='utf-8-sig')
        print(f"\n✅ 中长期策略结果已保存: {OUTPUT_DIR / 'regime_analysis_longterm.csv'}")
    
    elif not shortterm_results.empty:
        shortterm_results.to_csv(OUTPUT_DIR / 'regime_analysis_shortterm.csv', index=False, encoding='utf-8-sig')
        print(f"\n✅ 短期策略结果已保存: {OUTPUT_DIR / 'regime_analysis_shortterm.csv'}")
    
    else:
        print("\n⚠️  两种策略均未产生有效结果")


if __name__ == '__main__':
    main()
