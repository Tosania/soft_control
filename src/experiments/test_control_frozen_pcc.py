import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List, Dict

from src.core.tester import SoftRobotTester

# --- 绘图风格设置 ---
sns.set_theme(style="whitegrid", palette="bright", context="paper")
plt.rcParams["axes.unicode_minus"] = False

@dataclass
class DisturbanceEvent:
    type: str  # 'step' 或 'impulse'
    start_time: float  # (s)
    duration: float  # (s)
    force: np.ndarray  # [Fx, Fy, Fz] (N)

class FrozenPCCEvaluator:
    def __init__(self, xml_path: str, model_path: str = None, render: bool = False):
        self.xml_path = xml_path
        self.model_path = model_path

        init_mode = "hybrid" if model_path else "pcc"
        print(f"[Init] Initializing Tester with mode='{init_mode}' to ensure model loading...")
        self.tester = SoftRobotTester(
            xml_path=xml_path,
            model_path=model_path,
            mode=init_mode,
            render=render,
            video=False,
        )

        if model_path and not self.tester.model_loaded:
            print("[Warning] Model failed to load! Fallback to PCC only.")

        self.result_pcc_frozen = {}
        self.result_hybrid_frozen = {}

    def _run_single_pass(
        self,
        mode: str,
        duration: float,
        target_pos: list,
        events: List[DisturbanceEvent],
        freeze_time: float = 2.0
    ) -> Dict:
        """运行单次仿真：预热 -> 冻结 PCC -> 施加扰动"""
        print(f"\n   > Running Simulation in [{mode.upper()}] mode (Frozen PCC at {freeze_time}s)...")

        self.tester.mode = mode
        self.tester.reset()
        self.tester.set_target(target_pos)
        
        # 预热并稳定在目标点
        self.tester.stabilize_at(target_pos, tolerance=0.00, max_steps=800)

        history = {
            "time": [],
            "error": [],
            "force_mag": [],
            "cmd_pcc": [],
            "cmd_rl": [],
            "cmd_total": [],
        }

        start_time = self.tester.mj_data.time
        current_sim_time = 0
        pcc_frozen = False

        while current_sim_time < duration:
            current_sim_time = self.tester.mj_data.time - start_time

            # 触发冻结 PCC
            if current_sim_time >= freeze_time and not pcc_frozen:
                self.tester.freeze_pcc_control(True)
                pcc_frozen = True
                print(f"     => [Event] PCC Frozen at {current_sim_time:.2f}s")

            # 干扰调度
            current_force = np.zeros(3)
            for event in events:
                if event.type == "step":
                    if event.start_time <= current_sim_time <= (event.start_time + event.duration):
                        current_force += event.force
                elif event.type == "impulse":
                    if event.start_time <= current_sim_time <= (event.start_time + 0.05):
                        current_force += event.force

            self.tester.set_load(current_force)
            info = self.tester.step()

            # 记录数据
            history["time"].append(current_sim_time)
            history["error"].append(info["error"])
            history["force_mag"].append(np.linalg.norm(current_force))
            history["cmd_pcc"].append(info["cmd_pcc"].copy())
            history["cmd_rl"].append(info["cmd_rl"].copy())
            history["cmd_total"].append(info["cmd_total"].copy())

        return history

    def run_experiment(self, duration: float, target_pos: list, events: List[DisturbanceEvent]):
        print(f"[Evaluator] Start Comparative Experiment (Duration: {duration}s)")
        
        # 1. 运行纯 Frozen PCC (对照组)
        self.result_pcc_frozen = self._run_single_pass("pcc", duration, target_pos, events, freeze_time=2.0)

        # 2. 运行 Frozen PCC + RL (实验组)
        self.result_hybrid_frozen = self._run_single_pass("hybrid", duration, target_pos, events, freeze_time=2.0)

        print("[Evaluator] Experiment Finished. Closing tester.")
        self.tester.close()

    def plot_analysis(self, save_path="frozen_pcc_comparison.png"):
        """绘制极端对比图：冻结PCC无法恢复 vs RL主动调节"""
        time_axis = np.array(self.result_hybrid_frozen["time"])
        err_pcc = np.array(self.result_pcc_frozen["error"])
        err_hybrid = np.array(self.result_hybrid_frozen["error"])
        force = np.array(self.result_hybrid_frozen["force_mag"])
        
        cmd_rl = np.array(self.result_hybrid_frozen["cmd_rl"])
        cmd_pcc = np.array(self.result_hybrid_frozen["cmd_pcc"])
        cmd_total = np.array(self.result_hybrid_frozen["cmd_total"])
        
        colors = plt.cm.get_cmap("tab10", 8)
        actuator_labels = [f"Act {i+1}" for i in range(8)]

        fig, axes = plt.subplots(4, 1, figsize=(12, 18), sharex=True)

        # ==========================================
        # 图1: 末端误差对比
        # ==========================================
        ax1 = axes[0]
        ax1.plot(time_axis, err_pcc, color="gray", linestyle="--", linewidth=2.0, label="Frozen PCC (Open Loop)")
        ax1.plot(time_axis, err_hybrid, color="#d62728", linewidth=2.5, label="Frozen PCC + RL (Closed Loop)")

        ax1.set_title("1. Tracking Error: Frozen PCC vs Hybrid", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Error (m)", fontsize=12)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper right", framealpha=0.9)
        
        # 标记受力区域
        is_force = force > 1e-3
        ax1.fill_between(
            time_axis, ax1.get_ylim()[0], ax1.get_ylim()[1],
            where=is_force, color="orange", alpha=0.2, label="Impulse Disturbance"
        )
        # 标记冻结时间
        ax1.axvline(x=2.0, color='black', linestyle='-.', alpha=0.7, label="PCC Frozen")
        ax1.legend(loc="upper left")

        # ==========================================
        # 图2: RL 补偿输出
        # ==========================================
        ax2 = axes[1]
        ax2.set_title("2. RL Residual Output (Active Compensation)", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Delta Length (m)", fontsize=12)
        for i in range(8):
            ax2.plot(time_axis, cmd_rl[:, i], color=colors(i), linewidth=1.5, alpha=0.9, label=actuator_labels[i])
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.axvline(x=2.0, color='black', linestyle='-.', alpha=0.7)

        # ==========================================
        # 图3: PCC 基准输出 (被冻结)
        # ==========================================
        ax3 = axes[2]
        ax3.set_title("3. PCC Base Command (Frozen Constant)", fontsize=14, fontweight="bold")
        ax3.set_ylabel("Base Length (m)", fontsize=12)
        for i in range(8):
            ax3.plot(time_axis, cmd_pcc[:, i], color=colors(i), linewidth=1.5, alpha=0.9)
        ax3.autoscale(enable=True, axis="y", tight=False)
        ax3.grid(True, linestyle="--", alpha=0.5)
        ax3.axvline(x=2.0, color='black', linestyle='-.', alpha=0.7)

        # ==========================================
        # 图4: 总控制量
        # ==========================================
        ax4 = axes[3]
        ax4.set_title("4. Total Control Command (Frozen PCC + RL)", fontsize=14, fontweight="bold")
        ax4.set_ylabel("Total Length (m)", fontsize=12)
        ax4.set_xlabel("Time (s)", fontsize=14)
        for i in range(8):
            ax4.plot(time_axis, cmd_total[:, i], color=colors(i), linewidth=1.5, alpha=0.9, label=actuator_labels[i])
        ax4.grid(True, linestyle="--", alpha=0.5)
        ax4.axvline(x=2.0, color='black', linestyle='-.', alpha=0.7)
        ax4.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=8, fancybox=True, shadow=True, fontsize=10)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[Plot] Saved result to {save_path}")
        
        try:
            import matplotlib
            if matplotlib.get_backend().lower() != "agg":
                plt.show()
        except:
            pass
        finally:
            plt.close(fig)

if __name__ == "__main__":
    XML_FILE = "./assets/two_disks_uj.xml"
    MODEL_FILE = "./checkpoints/legacy_replica_v3/final_model.zip"

    evaluator = FrozenPCCEvaluator(XML_FILE, MODEL_FILE, render=True)

    # 干扰场景: 2s 冻结 PCC, 3s 加冲击力
    dist_scenario = [
        DisturbanceEvent(type="impulse", start_time=3.0, duration=0.05, force=np.array([8.0, 0, 0])),
    ]
    
    # 运行 6 秒
    evaluator.run_experiment(duration=6.0, target_pos=[0.4, 0.4, 0.7], events=dist_scenario)
    evaluator.plot_analysis(save_path="frozen_pcc_comparison.png")
