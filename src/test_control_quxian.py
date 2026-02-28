import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List, Dict
import matplotlib

# 使用非交互式后端，防止在无显示器的服务器上报错
# matplotlib.use("Agg")
# 引用底层驱动
from tester import SoftRobotTester

# --- 绘图风格设置 ---
sns.set_theme(style="whitegrid", palette="bright", context="paper")
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


@dataclass
class DisturbanceEvent:
    """定义单次干扰事件"""

    type: str  # 'step' (阶跃) 或 'impulse' (冲击)
    start_time: float  # 开始时间 (s)
    duration: float  # 持续时间 (s)
    force: np.ndarray  # 力向量 [Fx, Fy, Fz] (N)


class DisturbanceEvaluator:
    def __init__(self, xml_path: str, model_path: str = None, render: bool = False):
        self.xml_path = xml_path
        self.model_path = model_path

        # === [关键修复] ===
        # 即使我们想先测 PCC，如果提供了模型路径，也必须用 'hybrid' 初始化！
        # 否则 tester 内部会跳过模型加载，导致后续切换到 hybrid 时没有模型可用。
        init_mode = "hybrid" if model_path else "pcc"

        print(
            f"[Init] Initializing Tester with mode='{init_mode}' to ensure model loading..."
        )
        self.tester = SoftRobotTester(
            xml_path=xml_path,
            model_path=model_path,
            mode=init_mode,
            render=render,
            video=False,
        )

        # 检查模型是否加载成功
        if model_path and not self.tester.model_loaded:
            print("[Warning] Model failed to load! Fallback to PCC only.")

        # 存储两次运行的结果
        self.result_pcc = {}
        self.result_hybrid = {}

    def _run_single_pass(
        self,
        mode: str,
        duration: float,
        target_pos: list,
        events: List[DisturbanceEvent],
    ) -> Dict:
        """
        内部辅助函数：执行单次仿真运行
        """
        print(f"   > Running Simulation in [{mode.upper()}] mode...")

        # 1. 切换模式
        self.tester.mode = mode
        # 2. 重置环境
        self.tester.reset()
        # 3. 设置目标
        self.tester.set_target(target_pos)

        # 预热 (让机器人先到达目标附近，避免初始位置的巨大误差影响对比)
        # 注意：PCC 和 Hybrid 的预热行为可能略有不同，但为了公平对比抗扰动能力，
        # 我们让它们都从相对稳定的状态开始。
        self.tester.stabilize_at(target_pos, tolerance=0.00, max_steps=800)

        # 数据记录容器
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

        while current_sim_time < duration:
            current_sim_time = self.tester.mj_data.time - start_time

            # --- 干扰调度逻辑 ---
            current_force = np.zeros(3)
            for event in events:
                if event.type == "step":
                    if (
                        event.start_time
                        <= current_sim_time
                        <= (event.start_time + event.duration)
                    ):
                        current_force += event.force
                elif event.type == "impulse":
                    # 冲击仅持续 0.05s
                    if (
                        event.start_time
                        <= current_sim_time
                        <= (event.start_time + 0.05)
                    ):
                        current_force += event.force

            # 应用负载并步进
            self.tester.set_load(current_force)
            info = self.tester.step()

            # --- 记录数据 ---
            history["time"].append(current_sim_time)
            history["error"].append(info["error"])
            history["force_mag"].append(np.linalg.norm(current_force))
            history["cmd_pcc"].append(info["cmd_pcc"].copy())
            history["cmd_rl"].append(info["cmd_rl"].copy())
            history["cmd_total"].append(info["cmd_total"].copy())

        return history

    def run_experiment(
        self, duration: float, target_pos: list, events: List[DisturbanceEvent]
    ):
        """
        执行两次实验：一次 PCC，一次 Hybrid
        """
        print(f"[Evaluator] Start Comparative Experiment (Duration: {duration}s)")

        # 1. 运行 PCC 基准 (手动切换模式)
        self.result_pcc = self._run_single_pass("pcc", duration, target_pos, events)

        # 2. 运行 Hybrid (RL)
        # 如果模型没加载成功，这一步其实也是 PCC，但我们会尝试运行
        self.result_hybrid = self._run_single_pass(
            "hybrid", duration, target_pos, events
        )

        print("[Evaluator] Experiment Finished. Closing tester.")
        self.tester.close()

    def plot_analysis(self, save_path=None):
        """
        绘制对比图
        图1: PCC vs Hybrid 误差对比
        图2-4: Hybrid 模式下的控制详情
        """
        # 提取 Hybrid 的时间轴（假设两者时间同步）
        time_axis = np.array(self.result_hybrid["time"])

        # 提取误差数据
        err_pcc = np.array(self.result_pcc["error"])
        err_rl = np.array(self.result_hybrid["error"])

        # 提取力数据 (用于背景高亮)
        force = np.array(self.result_hybrid["force_mag"])

        # 提取 Hybrid 的控制量用于分析
        cmd_rl = np.array(self.result_hybrid["cmd_rl"])
        cmd_pcc = np.array(self.result_hybrid["cmd_pcc"])
        cmd_total = np.array(self.result_hybrid["cmd_total"])

        # 颜色设置
        colors = plt.cm.get_cmap("tab10", 8)
        actuator_labels = [f"Act {i+1}" for i in range(8)]

        fig, axes = plt.subplots(4, 1, figsize=(12, 18), sharex=True)

        # ==========================================
        # 图1: 末端误差对比 (Comparison)
        # ==========================================
        ax1 = axes[0]

        # 1.1 画 PCC 基准线 (灰色虚线)
        ax1.plot(
            time_axis,
            err_pcc,
            color="gray",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label="PCC Baseline",
        )

        # 1.2 画 Hybrid 曲线 (红色实线)
        ax1.plot(time_axis, err_rl, color="#d62728", linewidth=2.0, label="Hybrid (RL)")

        ax1.set_title(
            "1. Tracking Error Comparison: PCC vs Hybrid",
            fontsize=14,
            fontweight="bold",
        )
        ax1.set_ylabel("Error (m)", fontsize=12)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper right", framealpha=0.9)

        # 标记受力区域
        self._highlight_disturbances(ax1, time_axis, force)

        # ==========================================
        # 图2: RL 补偿输出 (Hybrid Run)
        # ==========================================
        ax2 = axes[1]
        ax2.set_title(
            "2. RL Residual Output (Hybrid Mode)", fontsize=14, fontweight="bold"
        )
        ax2.set_ylabel("Delta Length (m)", fontsize=12)

        for i in range(8):
            ax2.plot(
                time_axis,
                cmd_rl[:, i],
                color=colors(i),
                linewidth=1.5,
                alpha=0.9,
                label=actuator_labels[i],
            )
        ax2.grid(True, linestyle="--", alpha=0.5)

        # ==========================================
        # 图3: PCC 基准输出 (Hybrid Run)
        # ==========================================
        ax3 = axes[2]
        ax3.set_title(
            "3. PCC Base Command (Model-Based)", fontsize=14, fontweight="bold"
        )
        ax3.set_ylabel("Base Length (m)", fontsize=12)

        for i in range(8):
            ax3.plot(
                time_axis, cmd_pcc[:, i], color=colors(i), linewidth=1.5, alpha=0.9
            )

        ax3.autoscale(enable=True, axis="y", tight=False)
        ax3.grid(True, linestyle="--", alpha=0.5)

        # ==========================================
        # 图4: 总控制量 (Hybrid Run)
        # ==========================================
        ax4 = axes[3]
        ax4.set_title(
            "4. Total Control Command (PCC + RL)", fontsize=14, fontweight="bold"
        )
        ax4.set_ylabel("Total Length (m)", fontsize=12)
        ax4.set_xlabel("Time (s)", fontsize=14)

        for i in range(8):
            ax4.plot(
                time_axis,
                cmd_total[:, i],
                color=colors(i),
                linewidth=1.5,
                alpha=0.9,
                label=actuator_labels[i],
            )

        ax4.grid(True, linestyle="--", alpha=0.5)
        ax4.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=8,
            fancybox=True,
            shadow=True,
            fontsize=10,
        )

        # ==========================================
        # 保存与清理
        # ==========================================
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

    def _highlight_disturbances(self, ax, time, force):
        """仅在图1保留，用于标记时间点"""
        is_force = force > 1e-3
        ax.fill_between(
            time,
            ax.get_ylim()[0],
            ax.get_ylim()[1],
            where=is_force,
            color="gray",
            alpha=0.1,
            label="Disturbance Active",
        )

    def plot_paper_error_only(self, save_path="error_analysis_paper.png"):
        """
        专门导出适合论文的高质量误差对比图
        修复了 AttributeError: 'DisturbanceEvaluator' object has no attribute 'history'
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        # 1. 设置论文投稿风格 (更符合 IEEE/Elsevier 等期刊要求)
        sns.set_theme(style="ticks", context="paper")
        plt.rcParams.update(
            {
                "font.family": "serif",
                "mathtext.fontset": "stix",
                "axes.labelsize": 12,
                "legend.fontsize": 10,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
            }
        )

        # 检查数据是否存在
        if not self.result_hybrid or not self.result_pcc:
            print("错误: 数据缺失，请确保已运行 run_experiment()")
            return

        # 数据准备
        time_axis = np.array(self.result_hybrid["time"])
        err_hybrid = np.array(self.result_hybrid["error"])
        err_pcc = np.array(self.result_pcc["error"])
        force = np.array(self.result_hybrid["force_mag"])

        # 2. 创建画布 (宽度7英寸，高度3.8英寸，比例协调)
        fig, ax = plt.subplots(figsize=(7, 3.8), dpi=300)

        # 3. 绘制干扰背景 (阴影区) - 放在底层 (zorder=1)
        is_force = force > 0.05
        y_max = max(np.max(err_pcc), np.max(err_hybrid)) * 1.1
        ax.fill_between(
            time_axis,
            0,
            y_max,
            where=is_force,
            color="#E6E6E6",
            alpha=0.7,
            label="External Disturbance",
            zorder=1,
        )

        # 4. 绘制两条对比曲线
        # PCC 设为灰色虚线作为基准
        ax.plot(
            time_axis,
            err_pcc,
            color="#7F7F7F",
            linestyle="--",
            linewidth=1.2,
            label="Conventional PCC",
            zorder=2,
        )

        # Hybrid (本项目方法) 设为显眼的深蓝色或深红色实线
        ax.plot(
            time_axis,
            err_hybrid,
            color="#E31A1C",
            linestyle="-",
            linewidth=1.8,
            label="Hybrid Control (Ours)",
            zorder=3,
        )

        # 5. 精细化修饰
        ax.set_xlabel("Time (s)", fontweight="bold")
        ax.set_ylabel("Tracking Error (m)", fontweight="bold")

        # 移除顶部和右侧多余边框
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # 添加水平参考线
        ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)

        # 锁定范围
        ax.set_xlim(time_axis.min(), time_axis.max())
        ax.set_ylim(0, y_max)

        # 6. 图例美化 (放在右上角，带轻微背景色防止遮挡)
        ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="none")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"[*] 论文对比误差图已导出至: {save_path}")

        plt.close(fig)

    def plot_impulse_zoom(
        self, save_path="impulse_zoom_analysis.png", zoom_window=(4.5, 6.5)
    ):
        """
        专门针对第5秒冲击干扰的局部放大图
        zoom_window: 显示的时间范围，默认从 4.5s 到 6.5s
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        # 1. 风格设置
        sns.set_theme(style="ticks", context="paper")
        plt.rcParams.update(
            {
                "font.family": "serif",
                "mathtext.fontset": "stix",
                "axes.labelsize": 12,
                "legend.fontsize": 9,
            }
        )

        # 2. 提取并过滤数据
        t = np.array(self.result_hybrid["time"])
        mask = (t >= zoom_window[0]) & (t <= zoom_window[1])

        t_zoom = t[mask]
        err_pcc = np.array(self.result_pcc["error"])[mask]
        err_hybrid = np.array(self.result_hybrid["error"])[mask]
        force = np.array(self.result_hybrid["force_mag"])[mask]

        # 3. 创建画布 (比例建议更方一些，突出动态过程)
        fig, ax = plt.subplots(figsize=(5, 4), dpi=300)

        # 4. 绘制干扰发生的时间区间 (深色竖条表示冲击)
        is_force = force > 0.05
        y_max = max(np.max(err_pcc), np.max(err_hybrid)) * 1.1
        ax.fill_between(
            t_zoom,
            0,
            y_max,
            where=is_force,
            color="#636363",
            alpha=0.3,
            label="Impulse Event",
            zorder=1,
        )

        # 5. 绘制对比曲线
        # PCC 线：加粗虚线，展示巨大的波动
        ax.plot(
            t_zoom,
            err_pcc,
            color="#7F7F7F",
            linestyle="--",
            linewidth=1.2,
            label="Conventional PCC",
            zorder=2,
        )

        # Hybrid 线：实线，展示快速的收敛
        ax.plot(
            t_zoom,
            err_hybrid,
            color="#E31A1C",
            linestyle="-",
            linewidth=1.8,
            label="Hybrid (Ours)",
            zorder=3,
        )

        # 6. 精细化细节
        ax.set_xlabel("Time (s)", fontweight="bold")
        ax.set_ylabel("Error (m)", fontweight="bold")

        # 移除多余边框
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # 自动调整坐标轴，让对比最明显
        ax.set_xlim(zoom_window)
        ax.set_ylim(0, y_max)

        # 网格辅助线
        ax.grid(True, linestyle=":", alpha=0.6)

        # 图例：放在干扰较少的一侧
        ax.legend(loc="upper right", frameon=True, framealpha=0.9)

        # 在图上通过文字标注性能提升 (可选)
        # peak_pcc = np.max(err_pcc)
        # peak_hybrid = np.max(err_hybrid)
        # reduction = (peak_pcc - peak_hybrid) / peak_pcc * 100
        # ax.text(zoom_window[0]+0.1, y_max*0.9, f"Peak Error Reduced by {reduction:.1f}%",
        #         color="#E31A1C", weight='bold', fontsize=9)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"[*] 局部放大图（冲击响应）已导出至: {save_path}")

        plt.close(fig)


# ==========================================
# 运行入口
# ==========================================
if __name__ == "__main__":
    XML_FILE = "./source/two_disks_uj.xml"

    # ⚠️ 请确保这里填入了正确的 RL 模型路径
    MODEL_FILE = "./models/legacy_replica_v3/final_model.zip"

    evaluator = DisturbanceEvaluator(XML_FILE, MODEL_FILE, render=True)

    # 定义干扰场景
    dist_scenario = [
        DisturbanceEvent(
            type="impulse", start_time=3.0, duration=0.05, force=np.array([8.0, 0, 0])
        ),
        DisturbanceEvent(
            type="step", start_time=6.0, duration=3.0, force=np.array([0, 0, -7])
        ),
    ]
    evaluator.run_experiment(
        duration=10, target_pos=[0.4, 0.4, 0.7], events=dist_scenario
    )
    evaluator.plot_analysis(save_path="disturbance_analysis_comparison_fixed.png")
