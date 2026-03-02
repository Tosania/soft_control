import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.core.tester import SoftRobotTester
from src.core.adaptive_tester import AdaptiveSoftRobotTester

def run_comparison():
    """
    运行对比测试：分别基于同一目标点和相同扰动，
    对比 Normal Tester (固定 Scale) 和 Adaptive Tester (自适应 Scale) 的控制效果。
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "../../checkpoints/legacy_replica_v3/final_model.zip")
    xml_path = os.path.join(current_dir, "../../assets/two_disks_uj.xml")
    
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}. Running in PCC mode.")
        mode = "pcc"
    else:
        mode = "hybrid"

    # 测试参数设置
    target = np.array([0.3, 0.3, 0.9])
    simulation_steps = 1500
    disturbance_start = 500
    disturbance_duration = 200
    disturbance_force = np.array([1.5, 0.0, 0.0]) # 施加 1.5N 的 X 轴方向扰动

    print("=========================================")
    print("1. Running Normal Controller Test (Fixed Scale)")
    print("=========================================")
    # 正常固定 Scale 控制器测试
    normal_tester = SoftRobotTester(
        xml_path=xml_path,
        model_path=model_path,
        mode=mode,
        render=False, # 设置为 False 以便连续快速执行测试，如果想看动画可以改成 True
        video=False
    )
    
    normal_tester.reset()
    normal_tester.set_target(target)
    
    normal_error = []
    
    # 预热选项 (如果需要机器人从稳定状态开始，可以去除注释):
    # normal_tester.stabilize_at(target)
    
    try:
        for step in range(simulation_steps):
            if disturbance_start <= step < disturbance_start + disturbance_duration:
                normal_tester.set_load(disturbance_force)
            else:
                normal_tester.set_load(np.zeros(3))
                
            info = normal_tester.step()
            normal_error.append(info["error"])
            
            if step % 100 == 0:
                print(f"[Normal] Step {step:4d} | Error: {info['error']:.4f}m")
    finally:
        normal_tester.close()


    print("\n=========================================")
    print("2. Running Adaptive Controller Test (Dynamic Scale)")
    print("=========================================")
    # 自适应 Scale 控制器测试
    adaptive_tester = AdaptiveSoftRobotTester(
        xml_path=xml_path,
        model_path=model_path,
        mode=mode,
        render=False,
        video=False
    )
    
    adaptive_tester.reset()
    adaptive_tester.set_target(target)
    
    adaptive_error = []
    adaptive_scale = []
    
    # adaptive_tester.stabilize_at(target)
    
    try:
        for step in range(simulation_steps):
            if disturbance_start <= step < disturbance_start + disturbance_duration:
                adaptive_tester.set_load(disturbance_force)
            else:
                adaptive_tester.set_load(np.zeros(3))
                
            info = adaptive_tester.step()
            adaptive_error.append(info["error"])
            adaptive_scale.append(info.get("residual_scale", 0.0))
            
            if step % 100 == 0:
                print(f"[Adaptive] Step {step:4d} | Error: {info['error']:.4f}m | Scale: {info.get('residual_scale', 0.0):.4f}")
    finally:
        adaptive_tester.close()

    print("\nSimulation Finished. Plotting results...")

    # ==== 绘制结果图形 ====
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    time_steps_arr = range(simulation_steps)
    
    # 绘制误差对比曲线
    ax1.plot(time_steps_arr, normal_error, label="Normal Controller (Fixed Scale=0.08)", color="orange", linewidth=2, linestyle="--")
    ax1.plot(time_steps_arr, adaptive_error, label="Adaptive Controller", color="blue", linewidth=2)
    ax1.axvspan(disturbance_start, disturbance_start + disturbance_duration, color='red', alpha=0.2, label=f"Disturbance Active ({disturbance_force} N)")
    ax1.set_ylabel("Error (m)")
    ax1.set_title(f"Tracking Error Comparison for Target: {target}")
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # 绘制 Scale 对比曲线
    ax2.plot(time_steps_arr, [0.08] * simulation_steps, label="Normal Controller Scale (0.08)", color="orange", linewidth=2, linestyle="--")
    ax2.plot(time_steps_arr, adaptive_scale, label="Adaptive Controller Scale", color="green", linewidth=2)
    ax2.axvspan(disturbance_start, disturbance_start + disturbance_duration, color='red', alpha=0.2, label="Disturbance Active")
    ax2.set_ylabel("RL Residual Scale")
    ax2.set_xlabel("Simulation Steps")
    ax2.set_ylim(0, 0.25)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_comparison()
