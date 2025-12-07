"""
完整 GA 优化流程自动化脚本

流程：
1. 训练 2023 参数 → 保存 ga_best_params_2023.pkl
2. 训练 2024 参数 → 保存 ga_best_params_2024.pkl  
3. 滚动窗口回测 → 生成 Period 1/2 回测结果

无需手动修改配置文件，一键完成全流程。
"""

import os
import sys
import pickle
import subprocess
from pathlib import Path
from datetime import datetime

# ================================================================================
# 主流程
# ================================================================================

def run_ga_training(start_date: str, end_date: str, output_suffix: str) -> bool:
    """
    运行 GA 训练并保存参数
    
    Args:
        start_date: 训练开始日期
        end_date: 训练结束日期
        output_suffix: 输出文件后缀（如 '2023'）
    
    Returns:
        是否成功完成
    """
    print("\n" + "="*80)
    print(f"🧬 开始 GA 训练: {start_date} ~ {end_date}")
    print("="*80)
    
    # 动态修改 train_ga_params.py 的配置
    script_path = Path(__file__).parent / 'train_ga_params.py'
    
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换配置参数
    content = content.replace(
        "TRAIN_START_DATE = '2023-01-01'",
        f"TRAIN_START_DATE = '{start_date}'"
    )
    content = content.replace(
        "TRAIN_END_DATE = '2023-12-31'",
        f"TRAIN_END_DATE = '{end_date}'"
    )
    content = content.replace(
        "OUTPUT_SUFFIX = '2023'",
        f"OUTPUT_SUFFIX = '{output_suffix}'"
    )
    
    # 临时保存修改后的配置
    temp_script = Path(__file__).parent / f'_temp_ga_{output_suffix}.py'
    with open(temp_script, 'w', encoding='utf-8') as f:
        f.write(content)
    
    try:
        # 执行训练脚本
        result = subprocess.run(
            [sys.executable, str(temp_script)],
            cwd=Path(__file__).parent,
            capture_output=False,
            text=True
        )
        
        if result.returncode != 0:
            print(f"\n❌ {output_suffix} 训练失败，退出码: {result.returncode}")
            return False
        
        print(f"\n✅ {output_suffix} 训练完成!")
        return True
        
    finally:
        # 清理临时文件
        if temp_script.exists():
            temp_script.unlink()


def run_backtest_with_ga() -> bool:
    """运行滚动窗口回测"""
    print("\n" + "="*80)
    print("📊 开始滚动窗口回测")
    print("="*80)
    
    backtest_script = Path(__file__).parent / 'backtest_with_ga.py'
    
    result = subprocess.run(
        [sys.executable, str(backtest_script)],
        cwd=Path(__file__).parent,
        capture_output=False,
        text=True
    )
    
    if result.returncode != 0:
        print(f"\n❌ 回测失败，退出码: {result.returncode}")
        return False
    
    print(f"\n✅ 回测完成!")
    return True


def main():
    base_dir = Path(__file__).parent
    
    print("\n" + "🚀"*40)
    print("完整 GA 优化 + 回测流程自动化")
    print("🚀"*40)
    
    start_time = datetime.now()
    
    # ============================================================
    # 步骤 1: 训练 2023 参数
    # ============================================================
    success_2023 = run_ga_training('2023-01-01', '2023-12-31', '2023')
    if not success_2023:
        print("\n⚠️  2023 训练失败，终止流程")
        return
    
    params_2023 = base_dir / 'ga_best_params_2023.pkl'
    if not params_2023.exists():
        print(f"\n❌ 未找到 {params_2023.name}，训练可能未正常保存")
        return
    
    print(f"✓ 已保存: {params_2023.name}")
    
    # ============================================================
    # 步骤 2: 训练 2024 参数
    # ============================================================
    success_2024 = run_ga_training('2024-01-01', '2024-12-31', '2024')
    if not success_2024:
        print("\n⚠️  2024 训练失败，但 2023 参数已保存，可手动运行 backtest_with_ga.py")
        return
    
    params_2024 = base_dir / 'ga_best_params_2024.pkl'
    if not params_2024.exists():
        print(f"\n❌ 未找到 {params_2024.name}，训练可能未正常保存")
        return
    
    print(f"✓ 已保存: {params_2024.name}")
    
    # ============================================================
    # 步骤 3: 滚动窗口回测
    # ============================================================
    success_backtest = run_backtest_with_ga()
    if not success_backtest:
        print("\n⚠️  回测失败，但参数已保存，可手动运行 backtest_with_ga.py")
        return
    
    # ============================================================
    # 完成总结
    # ============================================================
    end_time = datetime.now()
    elapsed = end_time - start_time
    
    print("\n" + "="*80)
    print("🎉 全流程完成!")
    print("="*80)
    print(f"总耗时: {elapsed}")
    print("\n生成的文件:")
    print("  📦 参数文件:")
    print(f"    - {params_2023.name}")
    print(f"    - {params_2024.name}")
    print("  📊 回测结果:")
    print("    - backtest_results_GA2023_Period1_2024.csv")
    print("    - backtest_results_GA2024_Period2_2025.csv")
    print("  📈 进化日志:")
    print("    - ga_evolution_history_2023.csv")
    print("    - ga_evolution_history_2024.csv")
    print("\n下一步:")
    print("  1. 查看回测 CSV，对比 GA 参数与手动配置的效果")
    print("  2. 绘制 evolution_history 的收敛曲线")
    print("  3. 将最优参数复制到 backtest_final.py 作为新基线")


if __name__ == '__main__':
    main()
