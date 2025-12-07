"""
遗传算法自适应参数优化框架 (Genetic Algorithm for Strategy Tuning)

训练区间：2023-01-01 ~ 2023-12-31
验证区间：2024-02-05 ~ 2024-09-20 (Period 1)

适用场景：
- 板块/个股阈值优化
- 非凸、多峰值参数空间
- 滚动窗口 Walk-Forward 分析

输出：
- ga_best_params_{sector}.pkl: 各板块最优参数
- ga_evolution_history.csv: 进化过程记录
- ga_backtest_validation.csv: 样本外验证结果
"""

import os
import sys
import pickle
import random
import copy
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.append('.')
from strategies.triclass_core import TriclassStrategy, normalize_stock_code

# ================================================================================
# 1. 遗传算法超参数配置
# ================================================================================

# 训练期配置（修改此处以支持不同年份的滚动训练）
TRAIN_START_DATE = '2023-01-01'
TRAIN_END_DATE = '2023-12-31'
OUTPUT_SUFFIX = '2023'  # 输出文件后缀，如 ga_best_params_2023.pkl

GENE_BOUNDS = {
    'entry_up_threshold':  (0.50, 0.70, 0.01),  # (min, max, step)
    'entry_down_cap':      (0.20, 0.40, 0.01),
    'entry_margin':        (0.10, 0.30, 0.01),
    'add_up_threshold':    (0.60, 0.80, 0.01),
    'exit_down_threshold': (0.45, 0.60, 0.01),
}

POPULATION_SIZE = 50      # 种群大小
GENERATIONS = 20          # 进化代数
MUTATION_RATE = 0.35      # 变异概率（提高到 0.35 增加探索）
ELITISM_COUNT = 3         # 精英保留数量（降低到 3 减少同质化）
TOURNAMENT_SIZE = 3       # 锦标赛选择规模

# 早停配置(防止无意义迭代)
EARLY_STOP_PATIENCE = 5   # 最优解连续 5 代不变则提前终止

# 断点续训配置
ENABLE_CHECKPOINTS = True   # 是否启用断点保存
CHECKPOINT_INTERVAL = 3     # 每完成N个板块保存一次检查点

# 适应度函数权重
ALPHA_DRAWDOWN = 1.2      # 回撤惩罚系数（越大越保守）
PENALTY_LOW_TRADES = 60   # 交易次数 < 3 的惩罚
PENALTY_HEAVY_LOSS = 100  # 年化收益 < -15% 的惩罚

# ================================================================================
# 2. 板块与股票映射（与 backtest_final.py 保持一致）
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

# 板块分组用于进化（可根据需要调整）
SECTOR_GROUPS = {
    'alcohol': [],
    'chip': [],
    'new energy': [],
    'batteries': [],
    'automobile': [],
    'electric power': [],
    'education': [],
    'engineering machinery': [],
    'wind power equipment': [],
    'Photovoltaic equipment': [],
    'home appliance': [],
    'precious metals': [],
    'stock': [],
    'insurance': [],
}

# 自动填充
for code, sector in STOCK_CLASSIFICATION_MAP.items():
    if sector in SECTOR_GROUPS:
        SECTOR_GROUPS[sector].append(code)


# ================================================================================
# 3. 遗传算法核心类
# ================================================================================

class GeneticOptimizer:
    """遗传算法参数优化器"""
    
    def __init__(
        self,
        strategy_template: TriclassStrategy,
        train_data_dict: dict,
        sector_name: str,
        stock_list: list,
    ):
        self.strategy = strategy_template
        self.train_data = train_data_dict
        self.sector_name = sector_name
        self.stock_list = stock_list
        
        self.evolution_log = []  # 记录每代进化历史
        
    def generate_individual(self) -> dict:
        """生成一个随机个体（参数组合）"""
        individual = {}
        for key, (low, high, step) in GENE_BOUNDS.items():
            # 在范围内随机取值并量化到 step
            val = random.uniform(low, high)
            val = round(val / step) * step
            individual[key] = round(val, 3)
        
        # 强制约束：add_up >= entry_up + 0.05
        if individual['add_up_threshold'] <= individual['entry_up_threshold']:
            individual['add_up_threshold'] = round(
                individual['entry_up_threshold'] + 0.05, 3
            )
        
        return individual
    
    def fitness_function(self, config: dict) -> float:
        """
        适应度函数：在板块内所有股票上回测，综合评估
        
        Fitness = Avg(Annual_Return) - α × Avg(Max_Drawdown) - Penalties
        
        惩罚项：
        - 交易次数 < 3: -60
        - 年化收益 < -15%: -100
        """
        annual_returns = []
        max_drawdowns = []
        trade_counts = []
        
        # 临时替换策略的默认配置
        original_config = self.strategy.default_config.copy()
        self.strategy.default_config.update(config)
        
        for code in self.stock_list:
            norm_code = normalize_stock_code(code)
            if norm_code not in self.train_data:
                continue
            
            df = self.train_data[norm_code]
            if len(df) < 100:
                continue
            
            try:
                result = self.strategy.backtest_stock(
                    df, norm_code, initial_capital=10_000_000, include_details=False
                )
                
                if not result.get('error'):
                    annual_returns.append(result['annual_return'])
                    max_drawdowns.append(result['max_drawdown'])
                    trade_counts.append(result['num_trades'])
            except Exception as e:
                print(f"    ⚠️  {norm_code} 回测失败: {e}")
                continue
        
        # 恢复原配置
        self.strategy.default_config = original_config
        
        if not annual_returns:
            return -9999.0  # 无效个体
        
        avg_return = np.mean(annual_returns)
        avg_drawdown = np.mean(max_drawdowns)
        avg_trades = np.mean(trade_counts)
        
        # 计算惩罚
        penalty = 0.0
        if avg_trades < 3:
            penalty += PENALTY_LOW_TRADES
        if avg_return < -15:
            penalty += PENALTY_HEAVY_LOSS
        
        # 适应度 = 收益 - α×回撤 - 惩罚
        fitness = avg_return - ALPHA_DRAWDOWN * avg_drawdown - penalty
        
        return fitness
    
    def crossover(self, parent1: dict, parent2: dict) -> dict:
        """交叉操作：随机混合两个父代的基因"""
        child = {}
        for key in GENE_BOUNDS:
            r = random.random()
            if r < 0.4:
                child[key] = parent1[key]
            elif r < 0.8:
                child[key] = parent2[key]
            else:
                # 取平均
                child[key] = round((parent1[key] + parent2[key]) / 2, 3)
        return child
    
    def mutate(self, individual: dict) -> dict:
        """变异操作：对某个基因施加高斯扰动"""
        mutated = copy.deepcopy(individual)
        key = random.choice(list(GENE_BOUNDS.keys()))
        low, high, step = GENE_BOUNDS[key]
        
        # 高斯扰动
        sigma = (high - low) * 0.15
        delta = random.gauss(0, sigma)
        mutated[key] += delta
        
        # 边界截断并量化
        mutated[key] = max(low, min(high, mutated[key]))
        mutated[key] = round(mutated[key] / step) * step
        mutated[key] = round(mutated[key], 3)
        
        return mutated
    
    def tournament_selection(self, fitness_scores: list, k: int = TOURNAMENT_SIZE) -> dict:
        """锦标赛选择：从种群中随机抽取 k 个个体，选出最优者"""
        candidates = random.sample(fitness_scores, k)
        winner = max(candidates, key=lambda x: x[1])
        return winner[0]
    
    def run_evolution(self) -> dict:
        """执行遗传算法进化"""
        print(f"\n🧬 开始进化: {self.sector_name} (股票数: {len(self.stock_list)})")
        print(f"   种群规模: {POPULATION_SIZE}, 代数: {GENERATIONS}, 变异率: {MUTATION_RATE}")
        
        # 1. 初始化种群
        population = [self.generate_individual() for _ in range(POPULATION_SIZE)]
        
        global_best_individual = None
        global_best_fitness = -999999
        no_improvement_count = 0  # 早停计数器
        
        for gen in range(GENERATIONS):
            # 2. 计算适应度
            fitness_scores = []
            for ind in population:
                fit = self.fitness_function(ind)
                fitness_scores.append((ind, fit))
            
            # 3. 排序
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            current_best_ind = fitness_scores[0][0]
            current_best_fit = fitness_scores[0][1]
            
            # 更新全局最优
            if current_best_fit > global_best_fitness:
                global_best_fitness = current_best_fit
                global_best_individual = copy.deepcopy(current_best_ind)
                no_improvement_count = 0  # 重置早停计数
            else:
                no_improvement_count += 1  # 无改进，计数+1
            
            # 记录日志
            log_entry = {
                'sector': self.sector_name,
                'generation': gen + 1,
                'best_fitness': current_best_fit,
                'avg_fitness': np.mean([x[1] for x in fitness_scores]),
                'worst_fitness': fitness_scores[-1][1],
                'best_params': str(current_best_ind),
            }
            self.evolution_log.append(log_entry)
            
            print(
                f"   Gen {gen+1:2d}/{GENERATIONS} | "
                f"Best Fit: {current_best_fit:>7.2f} | "
                f"Avg Fit: {log_entry['avg_fitness']:>7.2f} | "
                f"Params: {current_best_ind}"
            )
            
            # 早停检查
            if no_improvement_count >= EARLY_STOP_PATIENCE:
                print(f"   ⚠️  连续 {EARLY_STOP_PATIENCE} 代无改进，提前终止进化")
                break
            
            # 4. 生成下一代
            next_generation = []
            
            # 精英保留
            for i in range(ELITISM_COUNT):
                next_generation.append(copy.deepcopy(fitness_scores[i][0]))
            
            # 选择、交叉、变异
            while len(next_generation) < POPULATION_SIZE:
                parent1 = self.tournament_selection(fitness_scores)
                parent2 = self.tournament_selection(fitness_scores)
                
                child = self.crossover(parent1, parent2)
                
                if random.random() < MUTATION_RATE:
                    child = self.mutate(child)
                
                # 约束检查
                if child['add_up_threshold'] <= child['entry_up_threshold']:
                    child['add_up_threshold'] = round(
                        child['entry_up_threshold'] + 0.05, 3
                    )
                
                next_generation.append(child)
            
            population = next_generation
        
        print(f"🎉 {self.sector_name} 进化完成! 最佳适应度: {global_best_fitness:.2f}")
        print(f"   最优参数: {global_best_individual}\n")
        
        return global_best_individual


# ================================================================================
# 4. 检查点管理工具
# ================================================================================

def save_checkpoint(base_dir: Path, best_configs: dict, all_logs: list, suffix: str):
    """保存中间检查点"""
    checkpoint_dir = base_dir / 'ga_checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint_file = checkpoint_dir / f'checkpoint_{suffix}.pkl'
    checkpoint_data = {
        'best_configs': best_configs,
        'evolution_logs': all_logs,
        'timestamp': datetime.now().isoformat(),
        'completed_sectors': list(best_configs.keys())
    }
    
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    print(f"  💾 检查点已保存: {checkpoint_file.name}")


def load_checkpoint(base_dir: Path, suffix: str) -> dict | None:
    """加载已有检查点(如果存在)"""
    checkpoint_file = base_dir / 'ga_checkpoints' / f'checkpoint_{suffix}.pkl'
    
    if not checkpoint_file.exists():
        return None
    
    try:
        with open(checkpoint_file, 'rb') as f:
            data = pickle.load(f)
        
        print(f"\n📥 发现检查点: {checkpoint_file.name}")
        print(f"   时间戳: {data['timestamp']}")
        print(f"   已完成板块: {', '.join(data['completed_sectors'])}")
        
        response = input("\n是否从检查点继续? (y/n): ").strip().lower()
        if response == 'y':
            return data
        else:
            print("   ⚠️  忽略检查点,从头开始训练")
            return None
            
    except Exception as e:
        print(f"   ⚠️  检查点加载失败: {e}")
        return None


# ================================================================================
# 5. 数据加载工具
# ================================================================================

def load_training_data(features_dir: Path, start_date: str, end_date: str) -> dict:
    """
    加载训练数据到内存（加速 GA 迭代）
    
    Args:
        features_dir: 特征文件目录
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
    
    Returns:
        {stock_code: DataFrame}
    """
    print(f"\n📂 加载训练数据: {start_date} ~ {end_date}")
    
    start_ts = pd.Timestamp(start_date) - pd.Timedelta(days=100)  # 预留历史窗口
    end_ts = pd.Timestamp(end_date)
    
    data_dict = {}
    feature_files = list(features_dir.glob('*_features.csv'))
    
    for file_path in feature_files:
        code = file_path.stem.replace('_features', '')
        
        try:
            df = pd.read_csv(file_path)
            df['day'] = pd.to_datetime(df['day'])
            df = df.set_index('day').sort_index()
            
            # 时间切片
            df_slice = df[(df.index >= start_ts) & (df.index <= end_ts)]
            
            if len(df_slice) >= 80:  # 至少需要 60 + 20 天
                data_dict[code] = df_slice
        except Exception as e:
            print(f"   ⚠️  {code} 加载失败: {e}")
            continue
    
    print(f"✓ 成功加载 {len(data_dict)} 只股票的训练数据\n")
    return data_dict


# ================================================================================
# 6. 主程序
# ================================================================================

def main():
    base_dir = Path(__file__).resolve().parent
    features_dir = base_dir.parent / 'features'
    
    print("\n" + "="*80)
    print("遗传算法策略参数优化 (Genetic Algorithm Parameter Tuning)")
    print("="*80)
    
    # ============================================================
    # Phase 1: 训练期进化 (2023 年数据)
    # ============================================================
    print(f"\n【阶段一】基于 {TRAIN_START_DATE} ~ {TRAIN_END_DATE} 数据进行参数进化")
    print("-" * 80)
    
    # 尝试加载检查点
    checkpoint = None
    if ENABLE_CHECKPOINTS:
        checkpoint = load_checkpoint(base_dir, OUTPUT_SUFFIX)
    
    train_data = load_training_data(features_dir, TRAIN_START_DATE, TRAIN_END_DATE)
    
    # 初始化基础策略模板
    base_strategy = TriclassStrategy(
        model_path=str(base_dir / 'model_triclass_alpha.pth'),
        scaler_path=str(base_dir / 'scaler_triclass.pkl'),
    )
    
    # 从检查点恢复或初始化
    if checkpoint:
        best_configs = checkpoint['best_configs']
        all_evolution_logs = checkpoint['evolution_logs']
        completed_sectors = set(checkpoint['completed_sectors'])
        print(f"\n✓ 已从检查点恢复 {len(completed_sectors)} 个板块的结果\n")
    else:
        best_configs = {}
        all_evolution_logs = []
        completed_sectors = set()
    
    # 对每个板块进行独立进化
    sector_count = 0
    for sector, stocks in SECTOR_GROUPS.items():
        # 跳过已完成的板块
        if sector in completed_sectors:
            print(f"⏭️  {sector}: 已完成(从检查点恢复),跳过\n")
            continue
        if len(stocks) < 3:  # 板块样本太少，跳过
            print(f"⚠️  {sector}: 股票数不足 ({len(stocks)}), 跳过\n")
            continue
        
        try:
            optimizer = GeneticOptimizer(
                strategy_template=base_strategy,
                train_data_dict=train_data,
                sector_name=sector,
                stock_list=stocks,
            )
            
            best_param = optimizer.run_evolution()
            best_configs[sector] = best_param
            all_evolution_logs.extend(optimizer.evolution_log)
            
            sector_count += 1
            
            # 定期保存检查点
            if ENABLE_CHECKPOINTS and sector_count % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(base_dir, best_configs, all_evolution_logs, OUTPUT_SUFFIX)
                
        except Exception as e:
            print(f"\n❌ {sector} 进化过程出错: {e}")
            print("   保存当前进度到检查点...")
            if ENABLE_CHECKPOINTS:
                save_checkpoint(base_dir, best_configs, all_evolution_logs, OUTPUT_SUFFIX)
            print(f"\n⚠️  可稍后重新运行脚本从检查点继续")
            raise  # 重新抛出异常以终止程序
    
    # 保存最终进化结果
    print("\n💾 保存最终进化结果...")
    
    # 保存最终检查点
    if ENABLE_CHECKPOINTS:
        save_checkpoint(base_dir, best_configs, all_evolution_logs, OUTPUT_SUFFIX)
    
    params_file = base_dir / f'ga_best_params_{OUTPUT_SUFFIX}.pkl'
    log_file = base_dir / f'ga_evolution_history_{OUTPUT_SUFFIX}.csv'
    
    with open(params_file, 'wb') as f:
        pickle.dump(best_configs, f)
    
    df_log = pd.DataFrame(all_evolution_logs)
    df_log.to_csv(log_file, index=False)
    
    print(f"✓ 参数已保存: {params_file.name}")
    print(f"✓ 进化日志已保存: {log_file.name}")
    
    print("\n🏆 最终进化结果（各板块最优参数）:")
    print("-" * 80)
    for sector, params in best_configs.items():
        print(f"{sector:25s} | {params}")
    
    # ============================================================
    # Phase 2: 样本外验证（可选，根据训练期自动推断验证期）
    # ============================================================
    # 根据训练年份确定验证期
    if OUTPUT_SUFFIX == '2023':
        val_start, val_end = '2024-02-05', '2024-09-20'
        val_label = '2024 Period 1'
    elif OUTPUT_SUFFIX == '2024':
        val_start, val_end = '2025-02-03', '2025-09-30'
        val_label = '2025 Period 2'
    else:
        print("\n⚠️  未定义验证期，跳过阶段二")
        print("="*80)
        print("✅ 遗传算法参数优化完成!")
        print("="*80)
        return
    
    print("\n" + "="*80)
    print(f"【阶段二】样本外验证 ({val_label}: {val_start} ~ {val_end})")
    print("="*80)
    
    try:
        validation_data = load_training_data(features_dir, val_start, val_end)
    except Exception as e:
        print(f"\n⚠️  验证数据加载失败: {e}")
        print("   参数已保存,可稍后手动运行 backtest_with_ga.py")
        print("="*80)
        print("✅ 遗传算法参数优化完成(跳过验证)!")
        print("="*80)
        return
    
    # 使用进化出的参数构建新策略
    try:
        validation_strategy = TriclassStrategy(
            model_path=str(base_dir / 'model_triclass_alpha.pth'),
            scaler_path=str(base_dir / 'scaler_triclass.pkl'),
            classification_configs=best_configs,
            stock_classification_map=STOCK_CLASSIFICATION_MAP,
        )
    except Exception as e:
        print(f"\n⚠️  验证策略初始化失败: {e}")
        print("   参数已保存,可稍后手动运行 backtest_with_ga.py")
        print("="*80)
        print("✅ 遗传算法参数优化完成(跳过验证)!")
        print("="*80)
        return
    
    validation_results = []
    
    for code, df in validation_data.items():
        norm_code = normalize_stock_code(code)
        
        try:
            result = validation_strategy.backtest_stock(
                df, norm_code, initial_capital=10_000_000, include_details=False
            )
            
            if not result.get('error'):
                validation_results.append(result)
                sector = STOCK_CLASSIFICATION_MAP.get(code, 'unknown')
                print(
                    f"  ✓ {code:8s} ({sector:20s}) | "
                    f"年化: {result['annual_return']:>7.2f}% | "
                    f"回撤: {result['max_drawdown']:>7.2f}% | "
                    f"交易: {result['num_trades']:>3d}"
                )
        except Exception as e:
            print(f"  ⚠️  {code} 验证失败: {e}")
            continue
    
    if validation_results:
        df_val = pd.DataFrame(validation_results)
        val_csv = base_dir / f'ga_backtest_validation_{OUTPUT_SUFFIX}.csv'
        df_val.to_csv(val_csv, index=False)
        
        print("\n" + "-"*80)
        print("样本外验证统计:")
        print(f"  成功回测股票数: {len(validation_results)}")
        print(f"  平均年化收益:   {df_val['annual_return'].mean():>8.2f}%")
        print(f"  中位年化收益:   {df_val['annual_return'].median():>8.2f}%")
        print(f"  平均最大回撤:   {df_val['max_drawdown'].mean():>8.2f}%")
        print(f"  正收益股票数:   {(df_val['annual_return'] > 0).sum()}")
        print(f"✓ 验证结果已保存: {val_csv.name}")
    
    print("\n" + "="*80)
    print("✅ 遗传算法参数优化完成!")
    print("="*80)


if __name__ == '__main__':
    main()
