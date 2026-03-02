import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.core.adaptive_tester import AdaptiveSoftRobotTester

def run_adaptive_test():
    """
    运行自适应控制器测试：
    前500步稳定，500-700步施加扰动，观察自适应 Scale 如何变化
    """
    # 路径配置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "../../checkpoints/legacy_replica_v3/final_model.zip")
    xml_path = os.path.join(current_dir, "../../assets/two_disks_uj.xml")
    
    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}. Running in PCC mode.")
        mode = "pcc"
    else:
        mode = "hybrid"

    # 初始化自适应测试器
    tester = AdaptiveSoftRobotTester(
        xml_path=xml_path,
        model_path=model_path,
        mode=mode,
        render=True,
        video=False
    )
    
    # 测试参数
    target = np.array([0.3, 0.3, 0.9])
    simulation_steps = 1500
    disturbance_start = 500
    disturbance_duration = 200
    disturbance_force = np.array([0,0, 0.0]) # 施加 1.5N 扰动
    
    print(f"=========================================")
    print(f"Starting Adaptive Control Test")
    print(f"Target: {target}")
    print(f"Will apply disturbance {disturbance_force}N at step {disturbance_start} for {disturbance_duration} steps")
    print(f"=========================================\n")
    
    tester.reset()
    tester.set_target(target)
    
    # 日志记录
    history_error = []
    history_scale = []
    
    try:
        for step in range(simulation_steps):
            # 施加外部扰动
            if disturbance_start <= step < disturbance_start + disturbance_duration:
                tester.set_load(disturbance_force)
            else:
                tester.set_load(np.zeros(3))
                
            # 控制步进
            info = tester.step()
            
            # 记录数据
            error = info["error"]
            scale = info.get("residual_scale", 0.0)
            history_error.append(error)
            history_scale.append(scale)
            
            if step % 100 == 0:
                print(f"Step {step:4d} | Error: {error:.4f}m | Adaptive Scale: {scale:.4f}")
                
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        tester.close()
        
    print("\nSimulation Finished. Plotting results...")

    # ==== 绘制结果图形 ====
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    time_steps = range(len(history_error))
    
    # 绘制误差曲线
    ax1.plot(time_steps, history_error, label="Tracking Error (m)", color="b")
    ax1.axvspan(disturbance_start, disturbance_start + disturbance_duration, color='red', alpha=0.2, label="Disturbance Active")
    ax1.set_ylabel("Error (m)")
    ax1.set_title("Adaptive Control Performance under Disturbance")
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # 绘制 Scale 曲线
    ax2.plot(time_steps, history_scale, label="RL Residual Scale (Adaptive)", color="green", linewidth=2)
    ax2.axvspan(disturbance_start, disturbance_start + disturbance_duration, color='red', alpha=0.2, label="Disturbance Active")
    ax2.set_ylabel("Scale Value")
    ax2.set_xlabel("Simulation Steps")
    ax2.set_ylim(0, 0.25) # 稍微大于 max_scale 保留空间
    ax2.axhline(y=0.08, color='k', linestyle=':', label="Default Base Scale (0.08)")
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_adaptive_test()
