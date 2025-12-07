"""
使用遗传算法优化的参数进行滚动窗口回测 (Walk-Forward Backtest with GA)

工作流程：
1. 加载 ga_best_params_2023.pkl（基于 2023 数据训练）
2. 应用到 2024-02-05 ~ 2024-09-20 (Period 1) 进行样本外验证
3. 加载 ga_best_params_2024.pkl（基于 2024 数据训练，需先运行 train_ga_params.py 2024 版本）
4. 应用到 2025-02-03 ~ 2025-09-30 (Period 2) 进行样本外验证

这种滚动窗口方法避免了未来信息泄露，更贴近真实交易场景。
"""

import os
import sys
import pickle
from pathlib import Path

import pandas as pd

sys.path.append('.')
from strategies.triclass_core import TriclassStrategy

# ================================================================================
# 板块映射（与 train_ga_params.py 保持一致）
# ================================================================================

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
    
    # 贵金属
    '600547': 'precious metals', '601899': 'precious metals',
    '600489': 'precious metals', '002155': 'precious metals',
    '600311': 'precious metals',
    
    # 券商
    '600030': 'stock', '601995': 'stock', '601688': 'stock',
    '600837': 'stock', '000776': 'stock', '002736': 'stock',
    '601066': 'stock', '600999': 'stock',
    
    # 保险
    '601318': 'insurance', '601628': 'insurance', '601601': 'insurance',
    '601336': 'insurance', '601319': 'insurance',
}


# ================================================================================
# 回测函数（复用 backtest_final.py 的逻辑）
# ================================================================================

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
    
    lookback_start = start_ts - pd.Timedelta(days=65)
    
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
            df = pd.read_csv(features_path / file)
            df['day'] = pd.to_datetime(df['day'])
            df = df.set_index('day').sort_index()
            
            df_test = df[(df.index >= lookback_start) & (df.index <= end_ts)]
            
            if len(df_test) < 100:
                print(f"  ⚠️  {stock_code}: 数据不足 ({len(df_test)} 天 < 100 天)")
                results.append({'stock': stock_code, 'error': True})
                continue
            
            result = strategy.backtest_stock(df_test, stock_code)
            results.append(result)
            
            if not result.get('error', False):
                print(
                    f"  ✓ {stock_code}: 资产 {result.get('final_asset', 0):,.2f} | "
                    f"总收益 {result.get('total_return', 0):.2f}% | "
                    f"年化 {result.get('annual_return', 0):.2f}% | "
                    f"最大回撤 {result.get('max_drawdown', 0):.2f}% | "
                    f"交易 {result.get('num_trades', 0)}"
                )
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


# ================================================================================
# 主程序
# ================================================================================

def main():
    base_dir = Path(__file__).resolve().parent
    features_dir = base_dir.parent / 'features'
    
    print("\n" + "="*80)
    print("滚动窗口回测 (Walk-Forward Backtest with GA-Optimized Parameters)")
    print("="*80)
    
    # ============================================================
    # 阶段一：使用 2023 训练的参数回测 2024 Period 1
    # ============================================================
    print("\n【阶段一】应用 2023 GA 参数 → 2024 Period 1 回测")
    print("-" * 80)
    
    params_2023_path = base_dir / 'ga_best_params_2023.pkl'
    
    if not params_2023_path.exists():
        print(f"❌ 未找到 {params_2023_path}")
        print("   请先运行: python train_ga_params.py (基于 2023 数据)")
        return
    
    with open(params_2023_path, 'rb') as f:
        ga_configs_2023 = pickle.load(f)
    
    print(f"✓ 加载 GA 参数: {params_2023_path}")
    print(f"   板块数: {len(ga_configs_2023)}")
    for sector, params in ga_configs_2023.items():
        print(f"   {sector:25s} | {params}")
    
    # 构建策略
    strategy_2024 = TriclassStrategy(
        model_path=str(base_dir / 'model_triclass_alpha.pth'),
        scaler_path=str(base_dir / 'scaler_triclass.pkl'),
        classification_configs=ga_configs_2023,
        stock_classification_map=STOCK_CLASSIFICATION_MAP,
    )
    
    # 回测 2024 Period 1
    results_2024_p1 = backtest_period(
        strategy_2024,
        str(features_dir),
        'GA2023_Period1_2024',
        '2024-02-05',
        '2024-09-20'
    )
    
    # ============================================================
    # 阶段二：使用 2024 训练的参数回测 2025 Period 2
    # ============================================================
    print("\n" + "="*80)
    print("【阶段二】应用 2024 GA 参数 → 2025 Period 2 回测")
    print("-" * 80)
    
    params_2024_path = base_dir / 'ga_best_params_2024.pkl'
    
    if not params_2024_path.exists():
        print(f"⚠️  未找到 {params_2024_path}")
        print("   需要先基于 2024 数据训练 GA 参数:")
        print("   1. 修改 train_ga_params.py 中训练日期为 '2024-01-01' ~ '2024-12-31'")
        print("   2. 修改输出文件名为 'ga_best_params_2024.pkl'")
        print("   3. 运行: python train_ga_params.py")
        print("\n   跳过阶段二，仅完成阶段一回测。")
        return
    
    with open(params_2024_path, 'rb') as f:
        ga_configs_2024 = pickle.load(f)
    
    print(f"✓ 加载 GA 参数: {params_2024_path}")
    print(f"   板块数: {len(ga_configs_2024)}")
    for sector, params in ga_configs_2024.items():
        print(f"   {sector:25s} | {params}")
    
    # 构建策略
    strategy_2025 = TriclassStrategy(
        model_path=str(base_dir / 'model_triclass_alpha.pth'),
        scaler_path=str(base_dir / 'scaler_triclass.pkl'),
        classification_configs=ga_configs_2024,
        stock_classification_map=STOCK_CLASSIFICATION_MAP,
    )
    
    # 回测 2025 Period 2
    results_2025_p2 = backtest_period(
        strategy_2025,
        str(features_dir),
        'GA2024_Period2_2025',
        '2025-02-03',
        '2025-09-30'
    )
    
    print("\n" + "="*80)
    print("✅ 滚动窗口回测完成!")
    print("="*80)
    print("\n生成的结果文件:")
    print("  - backtest_results_GA2023_Period1_2024.csv  (2023 参数 → 2024 验证)")
    print("  - backtest_results_GA2024_Period2_2025.csv  (2024 参数 → 2025 验证)")


if __name__ == '__main__':
    main()
