import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import time
import numpy as np
import pandas as pd
from typing import List, Dict
import multiprocessing
from functools import partial

from src.core.tester import SoftRobotTester

def run_single_direction_test(args):
    """
    独立进程运行函数。为了多进程能序列化，参数打包传入。
    args 包含: (xml_path, model_path, target_pos, force_name, force_vec, p_idx)
    """
    xml_path, model_path, target, f_name, f_vec, p_idx = args
    
    # 每个进程需要初始化自己的 Tester 环境 (因为 MuJoCo 的模拟状态不是线程安全的)
    tester_pcc = SoftRobotTester(xml_path=xml_path, model_path=model_path, mode="pcc", render=False, video=False)
    
    # 测试 PCC (开环)
    res_pcc = run_simulation_logic(tester_pcc, target.copy(), f_vec)
    tester_pcc.close()

    # 测试 Hybrid (闭环)
    tester_hybrid = SoftRobotTester(xml_path=xml_path, model_path=model_path, mode="hybrid", render=False, video=False)
    res_hybrid = run_simulation_logic(tester_hybrid, target.copy(), f_vec)
    tester_hybrid.close()

    improvement = max(0, (res_pcc["final_error"] - res_hybrid["final_error"]) / res_pcc["final_error"] * 100) if res_pcc["final_error"] > 0 else 0

    return {
        "Target Point": f"Pt {p_idx+1} ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})",
        "Force": f_name,
        "Baseline Final Err (m)": res_pcc["final_error"],
        "RL Final Err (m)": res_hybrid["final_error"],
        "Improvement (%)": improvement,
        "RL Peak Err (m)": res_hybrid["peak_error"]
    }

def run_simulation_logic(tester, target_pos, force_vec, duration=6.0, freeze_time=2.0, impulse_time=3.0):
    tester.reset()
    tester.set_target(target_pos)
    tester.stabilize_at(target_pos, tolerance=0.01, max_steps=800)

    start_time = tester.mj_data.time
    current_sim_time = 0
    pcc_frozen = False
    
    errors = []
    times = []

    while current_sim_time < duration:
        current_sim_time = tester.mj_data.time - start_time

        if current_sim_time >= freeze_time and not pcc_frozen:
            tester.freeze_pcc_control(True)
            pcc_frozen = True

        current_force = np.zeros(3)
        if impulse_time <= current_sim_time <= (impulse_time + 0.05):
            current_force = force_vec

        tester.set_load(current_force)
        info = tester.step()

        errors.append(info["error"])
        times.append(current_sim_time)

    errors = np.array(errors)
    times = np.array(times)
    
    post_impulse_mask = times >= impulse_time
    post_impulse_errors = errors[post_impulse_mask]
    
    if len(post_impulse_errors) > 0:
        peak_error = np.max(post_impulse_errors)
        final_mask = times >= (duration - 0.5)
        final_error = np.mean(errors[final_mask]) if np.any(final_mask) else post_impulse_errors[-1]
    else:
        peak_error = errors[-1]
        final_error = errors[-1]

    return {"peak_error": peak_error, "final_error": final_error}

def run_batch_evaluation_parallel(xml_path, model_path, num_points=5):
    print(f"[Evaluator] Starting PARALLEL Batch Evaluation with {num_points} random target points...\n")
    
    force_magnitude = 8.0
    force_configs = {
        "+X": np.array([force_magnitude, 0, 0]),
        "-X": np.array([-force_magnitude, 0, 0]),
        "+Y": np.array([0, force_magnitude, 0]),
        "-Y": np.array([0, -force_magnitude, 0])
    }

    np.random.seed(42)
    target_points = []
    for _ in range(num_points):
        x = np.random.uniform(-0.4, 0.4)
        y = np.random.uniform(-0.4, 0.4)
        z = np.random.uniform(0.5, 0.9)
        target_points.append(np.array([x, y, z]))

    tasks = []
    for p_idx, target in enumerate(target_points):
        for f_name, f_vec in force_configs.items():
            tasks.append((xml_path, model_path, target, f_name, f_vec, p_idx))

    # 使用所有可用的 CPU 核心，保留一个以免卡死系统
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"[Parallel] Using {num_cores} logical cores to run {len(tasks)} tasks...")

    # 开始并行执行
    start_wall_time = time.time()
    with multiprocessing.Pool(processes=num_cores) as pool:
        # 使用 pool.map，这会阻塞直到所有任务完成，并按返回顺序收集结果
        results = pool.map(run_single_direction_test, tasks)
    end_wall_time = time.time()
    
    print(f"\n[Parallel] All tasks completed in {end_wall_time - start_wall_time:.2f} seconds.")

    # 整理输出表格
    df = pd.DataFrame(results)
    
    print("\n\n" + "="*80)
    print("🚀 FROZEN PCC RESIDUAL RL ROBUSTNESS EVALUATION REPORT")
    print("="*80)
    
    try:
        # 尝试使用 tabulate 以 Markdown 格式化
        markdown_table = df.to_markdown(index=False, floatfmt=".4f")
        print(markdown_table)
    except ImportError:
        # 如果依然没有 tabulate，回退到字符串表示 (不报错)
        print("[Note] 'tabulate' package not found. Showing raw textual table.")
        # pandas 自带的字符串打印也足够清晰
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.float_format', '{:.4f}'.format)
        print(df.to_string(index=False))
    
    print("\n\n--- Summary Statistics ---")
    avg_baseline = df["Baseline Final Err (m)"].mean()
    avg_rl = df["RL Final Err (m)"].mean()
    avg_improvement = df["Improvement (%)"].mean()
    print(f"Average Baseline Error:     {avg_baseline:.4f} m (Open loop drift after impulse)")
    print(f"Average RL Recovered Error: {avg_rl:.4f} m (Active compensation performance)")
    print(f"Average Improvement:        {avg_improvement:.2f}%")
    print("="*80)
    
    df.to_csv("frozen_pcc_batch_results.csv", index=False)
    print("\n[+] Results saved to frozen_pcc_batch_results.csv")

if __name__ == "__main__":
    XML_FILE = "./assets/two_disks_uj.xml"
    MODEL_FILE = "./checkpoints/legacy_replica_v3/final_model.zip"
    
    # 强制要求 spawn 模式（避免 MuJoCo 在 fork 下的内部 OpenGL 上下文冲突）
    multiprocessing.set_start_method('spawn')
    run_batch_evaluation_parallel(XML_FILE, MODEL_FILE, num_points=5)
