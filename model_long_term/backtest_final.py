"""
三分类策略完整回测框架
在两个时段分别回测：
  Period 1: 2024-02-05 至 2024-09-20
  Period 2: 2025-02-03 至 2025-09-30
"""
import os
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings('ignore')

sys.path.append('.')
from strategies.triclass_core import TriclassStrategy


STOCK_CLASSIFICATION_MAP = {
    '000858': 'alcohol',
    '600519': 'alcohol',
    '002304': 'alcohol',
    '000568': 'alcohol',
    '603369': 'alcohol',
    '603589': 'alcohol',
    '603198': 'alcohol',
    '603919': 'alcohol',
    '603986': 'chip',
    '688981': 'chip',
    '002371': 'chip',
    '600703': 'chip',
    '603501': 'chip',
    '688187': 'chip',
    '688008': 'chip',
    '300661': 'chip',
    '300223': 'chip',
    '300782': 'chip',
    '002049': 'chip',
    '300373': 'chip',
    '300346': 'chip',
    '300567': 'chip',
    '300458': 'chip',
    '002812': 'new energy',
    '300014': 'batteries',
    '002460': 'new energy',
    '300450': 'new energy',
    '300750': 'batteries',
    '002466': 'batteries',
    '603659': 'batteries',
    '002594': 'automobile',
    '601633': 'automobile',
    '600104': 'automobile',
    '000625': 'automobile',
    '601238': 'automobile',
    '002708': 'automobile',
    '600900': 'electric power',
    '003816': 'electric power',
    '601985': 'electric power',
    '600011': 'electric power',
    '600023': 'electric power',
    '000993': 'electric power',
    '300359': 'education',
    '002261': 'education',
    '600661': 'education',
    '002315': 'education',
    '603877': 'education',
    '002563': 'education',
    '002291': 'education',
    '002425': 'education',
    '002569': 'education',
    '000157': 'engineering machinery',
    '000425': 'engineering machinery',
    '600031': 'engineering machinery',
    '601100': 'engineering machinery',
    '002097': 'engineering machinery',
    '002202': 'wind power equipment',
    '601615': 'wind power equipment',
    '300443': 'wind power equipment',
    '002531': 'wind power equipment',
    '603606': 'wind power equipment',
    '601012': 'Photovoltaic equipment',
    '300274': 'Photovoltaic equipment',
    '002459': 'Photovoltaic equipment',
    '603806': 'Photovoltaic equipment',
    '688599': 'Photovoltaic equipment',
    '300118': 'Photovoltaic equipment',
    '002242': 'home appliance',
    '603486': 'home appliance',
    '002508': 'home appliance',
    '002032': 'home appliance',
    '603355': 'home appliance',
    '600547': 'precious metals',
    '601899': 'precious metals',
    '600489': 'precious metals',
    '002155': 'precious metals',
    '600311': 'precious metals',
    '600030': 'stock',
    '601995': 'stock',
    '601688': 'stock',
    '600837': 'stock',
    '000776': 'stock',
    '002736': 'stock',
    '601066': 'stock',
    '600999': 'stock',
    '601318': 'insurance',
    '601628': 'insurance',
    '601601': 'insurance',
    '601336': 'insurance',
    '601319': 'insurance',
}



def backtest_period(strategy, features_dir, period_name, start_date, end_date):
    """
    在指定时段进行批量回测
    """
    print(f"\n{'='*80}")
    print(f"回测时段: {period_name}")
    print(f"日期范围: {start_date} 至 {end_date}")
    print(f"{'='*80}")
    
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    
    # 需要在 start_date 之前至少60天的数据用于特征计算
    # 所以实际加载的数据应该从 start_date - 60天 开始
    lookback_start = start_ts - pd.Timedelta(days=65)  # 额外留5天余量
    
    features_path = Path(features_dir)
    if not features_path.exists():
        raise FileNotFoundError(f"未找到特征目录: {features_path}")

    feature_files = sorted([f.name for f in features_path.glob('*_features.csv')])
    results = []
    
    for idx, file in enumerate(feature_files):
        stock_code = file.replace('_features.csv', '')
        
        if idx % 20 == 0:
            print(f"进度: {idx}/{len(feature_files)}")
        
        try:
            # 加载数据
            df = pd.read_csv(features_path / file)
            df['day'] = pd.to_datetime(df['day'])
            df = df.set_index('day').sort_index()
            
            # 保留 start_date 前60天 到 end_date 的数据（为了回测时有足够的历史数据）
            df_test = df[(df.index >= lookback_start) & (df.index <= end_ts)]
            
            if len(df_test) < 100:  # 需要至少60天历史 + 回测期间
                print(f"  ⚠️  {stock_code}: 数据不足 ({len(df_test)} 天 < 100 天)")
                results.append({'stock': stock_code, 'error': True})
                continue
            
            # 回测
            result = strategy.backtest_stock(df_test, stock_code)
            results.append(result)
            if not result.get('error', False):
                print(f"  ✓ {stock_code}: 资产 {result.get('final_asset', 0):,.2f} | 总收益 {result.get('total_return', 0):.2f}% | 年化 {result.get('annual_return', 0):.2f}% | 最大回撤 {result.get('max_drawdown', 0):.2f}% | 交易 {result.get('num_trades', 0)}")
            else:
                print(f"  ⚠️ {stock_code}: 回测失败或数据不足")
            
        except Exception as e:
            print(f"  ❌ {stock_code}: {str(e)[:50]}")
            results.append({'stock': stock_code, 'error': True})
    
    # 统计
    valid_results = [r for r in results if not r.get('error', False)]
    
    if valid_results:
        df_results = pd.DataFrame(valid_results)
        
        print(f"\n{'统计结果':-^80}")
        print(f"总股票数: {len(feature_files)}")
        print(f"成功回测: {len(valid_results)}")
        print(f"\n年化收益率:")
        print(f"  平均: {df_results['annual_return'].mean():>8.2f}%")
        print(f"  中位数: {df_results['annual_return'].median():>8.2f}%")
        print(f"  最高: {df_results['annual_return'].max():>8.2f}%")
        print(f"  最低: {df_results['annual_return'].min():>8.2f}%")
        print(f"  正收益数: {(df_results['annual_return'] > 0).sum()}")
        
        print(f"\n最大回撤:")
        print(f"  平均: {df_results['max_drawdown'].mean():>8.2f}%")
        print(f"  中位数: {df_results['max_drawdown'].median():>8.2f}%")
        print(f"  最差: {df_results['max_drawdown'].min():>8.2f}%")
        
        print(f"\n交易统计:")
        print(f"  平均交易数: {df_results['num_trades'].mean():>8.1f}")
        
        # 保存结果
        csv_name = f"backtest_results_{period_name.replace(' ', '_')}.csv"
        df_results.to_csv(csv_name, index=False)
        print(f"\n✓ 结果已保存到 {csv_name}")
    
    return results


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent
    features_dir = base_dir.parent / 'features'

    classification_profiles = {
        'alcohol': 'defensive',
        'electric power': 'defensive',
        'home appliance': 'defensive',
        'insurance': 'defensive',
        'automobile': 'high_vol',
        'education': 'growth',
        'engineering machinery': 'growth',
        'stock': 'growth',
        'chip': 'high_vol',
        'new energy': 'high_vol',
        'batteries': 'high_vol',
        'Photovoltaic equipment': 'high_vol',
        'wind power equipment': 'high_vol',
        'precious metals': 'high_vol',
    }

    base_loose = {
        'entry_up_threshold': 0.52,
        'entry_down_cap': 0.30,
        'entry_margin': 0.17,
        'add_up_threshold': 0.64,
        'exit_down_threshold': 0.53,
    }
    base_mid = {
        'entry_up_threshold': 0.53,
        'entry_down_cap': 0.30,
        'entry_margin': 0.17,
        'add_up_threshold': 0.64,
        'exit_down_threshold': 0.53,
    }

    profile_configs_loose = {
        'defensive': {
            'entry_up_threshold': 0.51,
            'entry_down_cap': 0.32,
            'entry_margin': 0.16,
            'add_up_threshold': 0.63,
            'exit_down_threshold': 0.52,
        },
        'growth': {
            'entry_up_threshold': 0.56,
            'entry_down_cap': 0.29,
            'entry_margin': 0.18,
            'add_up_threshold': 0.66,
            'exit_down_threshold': 0.52,
        },
        'high_vol': {
            'entry_up_threshold': 0.62,
            'entry_down_cap': 0.26,
            'entry_margin': 0.21,
            'add_up_threshold': 0.72,
            'exit_down_threshold': 0.50,
        },
    }

    profile_configs_mid = {
        'defensive': {
            'entry_up_threshold': 0.52,
            'entry_down_cap': 0.31,
            'entry_margin': 0.17,
            'add_up_threshold': 0.64,
            'exit_down_threshold': 0.52,
        },
        'growth': {
            'entry_up_threshold': 0.57,
            'entry_down_cap': 0.28,
            'entry_margin': 0.19,
            'add_up_threshold': 0.67,
            'exit_down_threshold': 0.52,
        },
        'high_vol': {
            'entry_up_threshold': 0.64,
            'entry_down_cap': 0.25,
            'entry_margin': 0.22,
            'add_up_threshold': 0.74,
            'exit_down_threshold': 0.50,
        },
    }

    classification_configs_loose = {}
    classification_configs_mid = {}
    for cls in classification_profiles:
        profile = classification_profiles[cls]
        classification_configs_loose[cls] = dict(profile_configs_loose[profile])
        classification_configs_mid[cls] = dict(profile_configs_mid[profile])

    classification_configs_loose['high_volatility'] = dict(profile_configs_loose['high_vol'])
    classification_configs_mid['high_volatility'] = dict(profile_configs_mid['high_vol'])

    sector_overrides_loose = {
        'alcohol': dict(entry_up_threshold=0.53, entry_margin=0.18, entry_down_cap=0.31, exit_down_threshold=0.51),
        'home appliance': dict(entry_up_threshold=0.58, entry_margin=0.21, entry_down_cap=0.28, add_up_threshold=0.68, exit_down_threshold=0.50),
        'engineering machinery': dict(entry_up_threshold=0.57, entry_margin=0.20, entry_down_cap=0.28, add_up_threshold=0.68, exit_down_threshold=0.51),
        'automobile': dict(entry_up_threshold=0.66, entry_margin=0.24, entry_down_cap=0.22, add_up_threshold=0.78, exit_down_threshold=0.47),
        'new energy': dict(entry_up_threshold=0.66, entry_margin=0.24, entry_down_cap=0.22, add_up_threshold=0.78, exit_down_threshold=0.47),
        'batteries': dict(entry_up_threshold=0.66, entry_margin=0.24, entry_down_cap=0.22, add_up_threshold=0.78, exit_down_threshold=0.47),
        'Photovoltaic equipment': dict(entry_up_threshold=0.66, entry_margin=0.24, entry_down_cap=0.22, add_up_threshold=0.78, exit_down_threshold=0.47),
        'education': dict(add_up_threshold=0.68, exit_down_threshold=0.52),
    }

    sector_overrides_mid = {
        'alcohol': dict(entry_up_threshold=0.54, entry_margin=0.18, entry_down_cap=0.30, exit_down_threshold=0.51),
        'home appliance': dict(entry_up_threshold=0.58, entry_margin=0.21, entry_down_cap=0.28, add_up_threshold=0.69, exit_down_threshold=0.50),
        'engineering machinery': dict(entry_up_threshold=0.59, entry_margin=0.21, entry_down_cap=0.27, add_up_threshold=0.69, exit_down_threshold=0.51),
        'automobile': dict(entry_up_threshold=0.66, entry_margin=0.24, entry_down_cap=0.22, add_up_threshold=0.78, exit_down_threshold=0.47),
        'new energy': dict(entry_up_threshold=0.66, entry_margin=0.24, entry_down_cap=0.22, add_up_threshold=0.78, exit_down_threshold=0.47),
        'batteries': dict(entry_up_threshold=0.66, entry_margin=0.24, entry_down_cap=0.22, add_up_threshold=0.78, exit_down_threshold=0.47),
        'Photovoltaic equipment': dict(entry_up_threshold=0.66, entry_margin=0.24, entry_down_cap=0.22, add_up_threshold=0.78, exit_down_threshold=0.47),
        'education': dict(entry_up_threshold=0.58, entry_margin=0.20, add_up_threshold=0.69, exit_down_threshold=0.52),
    }

    for sector, overrides in sector_overrides_loose.items():
        if sector in classification_configs_loose:
            classification_configs_loose[sector].update(overrides)
    for sector, overrides in sector_overrides_mid.items():
        if sector in classification_configs_mid:
            classification_configs_mid[sector].update(overrides)

    stock_configs_common = {
        '002708': dict(entry_up_threshold=0.70, entry_margin=0.25, entry_down_cap=0.22, add_up_threshold=0.80, exit_down_threshold=0.46),
        '002261': dict(entry_up_threshold=0.62, entry_margin=0.22, entry_down_cap=0.25, add_up_threshold=0.74, exit_down_threshold=0.50),
        '300359': dict(entry_up_threshold=0.62, entry_margin=0.22, entry_down_cap=0.25, add_up_threshold=0.74, exit_down_threshold=0.50),
        '002459': dict(entry_up_threshold=0.68, entry_margin=0.23, entry_down_cap=0.22, add_up_threshold=0.78, exit_down_threshold=0.48),
        '300450': dict(entry_up_threshold=0.68, entry_margin=0.23, entry_down_cap=0.22, add_up_threshold=0.78, exit_down_threshold=0.48),
        '300782': dict(entry_up_threshold=0.68, entry_margin=0.23, entry_down_cap=0.22, add_up_threshold=0.78, exit_down_threshold=0.48),
        '603198': dict(entry_up_threshold=0.68, entry_margin=0.23, entry_down_cap=0.22, add_up_threshold=0.78, exit_down_threshold=0.48),
    }

    configs = [
        # (
        #     'cfg_routed_loose_margin017_exit053',
        #     dict(
        #         default_config=base_loose,
        #         classification_configs=classification_configs_loose,
        #         stock_configs=stock_configs_common,
        #     ),
        # ),
        (
            'cfg_routed_mid_entry053_exit053',
            dict(
                default_config=base_mid,
                classification_configs=classification_configs_mid,
                stock_configs=stock_configs_common,
            ),
        ),
    ]

    for cfg_name, cfg in configs:
        print(f"\n{'#'*80}")
        print(f"运行配置: {cfg_name}")
        print(f"默认参数: {cfg['default_config']}")
        print(f"板块覆盖: {len(cfg.get('classification_configs', {}))} 个")
        print(f"{'#'*80}\n")

        strategy = TriclassStrategy(
            'model_triclass_alpha.pth',
            'scaler_triclass.pkl',
            default_config=cfg['default_config'],
            stock_configs=cfg.get('stock_configs'),
            classification_configs=cfg.get('classification_configs'),
            stock_classification_map=STOCK_CLASSIFICATION_MAP,
        )

        results_1 = backtest_period(
            strategy,
            features_dir,
            f'{cfg_name}_Period_1_2024',
            '2024-02-05',
            '2024-09-20'
        )

        results_2 = backtest_period(
            strategy,
            features_dir,
            f'{cfg_name}_Period_2_2025',
            '2025-02-03',
            '2025-09-30'
        )

    print(f"\n{'='*80}")
    print("✓ 回测完成 (全部配置)")
    print(f"{'='*80}")
