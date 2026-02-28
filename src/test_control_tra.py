import time
import numpy as np
import mujoco
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tester import SoftRobotTester

# --- 绘图风格设置 ---
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = "sans-serif"


# ==========================================
# 1. 增强型轨迹生成器 (新增 Square)
# ==========================================
class TrajectoryPlanner:
    def __init__(self, traj_type, center_z=0.52, radius=0.15, period=10.0):
        self.type = traj_type
        self.center_z = center_z
        self.radius = radius
        self.period = period

    def get_target(self, t):
        phase = (t % self.period) / self.period

        if self.type == "Square":
            # 将相位 0~1 映射到正方形的四条边
            # 设正方形顶点为 (R, -R) -> (R, R) -> (-R, R) -> (-R, -R) -> (R, -R)
            # 逆时针运动，起点在右下角
            p = phase * 4.0
            r = self.radius

            if p < 1.0:  # 第一边：右侧边，向上移动 (+Y)
                local_p = p
                x = r
                y = -r + local_p * (2 * r)
            elif p < 2.0:  # 第二边：上侧边，向左移动 (-X)
                local_p = p - 1.0
                x = r - local_p * (2 * r)
                y = r
            elif p < 3.0:  # 第三边：左侧边，向下移动 (-Y)
                local_p = p - 2.0
                x = -r
                y = r - local_p * (2 * r)
            else:  # 第四边：下侧边，向右移动 (+X)
                local_p = p - 3.0
                x = -r + local_p * (2 * r)
                y = -r

            z = self.center_z

        elif self.type == "Rose":
            theta = 2 * np.pi * phase
            k = 3
            r = self.radius * np.cos(k * theta)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = self.center_z

        elif self.type == "Circle":
            theta = 2 * np.pi * phase
            x = self.radius * np.cos(theta)
            y = self.radius * np.sin(theta)
            z = self.center_z

        else:
            x, y, z = 0, 0, self.center_z

        return np.array([x, y, z])


# ==========================================
# 2. 扰动调度器 (复杂负载逻辑)
# ==========================================
class DisturbanceScheduler:
    """
    管理复杂的负载事件：每次扰动包含一个短时高幅值的'冲击'，
    随后紧接一个较长时间低幅值的'阶跃'。
    """

    def __init__(self, events):
        """
        events: list of dict, 每个字典包含:
          - start_time: 开始时间
          - impulse_vec: 冲击力向量 (N)
          - step_vec: 阶跃力向量 (N)
          - impulse_dur: 冲击持续时间 (s)
          - step_dur: 阶跃持续时间 (s)
        """
        self.events = events
        self.active_intervals = []  # 用于绘图记录 (start, end)

        # 预计算绘图区间
        for ev in self.events:
            total_dur = ev["impulse_dur"] + ev["step_dur"]
            self.active_intervals.append(
                (ev["start_time"], ev["start_time"] + total_dur)
            )

    def get_force(self, t):
        current_force = np.zeros(3)

        for ev in self.events:
            t_local = t - ev["start_time"]

            # 检查是否在事件范围内
            if t_local >= 0:
                # 阶段1: 冲击 (Impulse)
                if t_local < ev["impulse_dur"]:
                    current_force = np.array(ev["impulse_vec"])
                    break  # 同一时间只允许一个主扰动

                # 阶段2: 阶跃 (Step)
                elif t_local < (ev["impulse_dur"] + ev["step_dur"]):
                    current_force = np.array(ev["step_vec"])
                    break

        return current_force


# ==========================================
# 3. 仿真运行逻辑
# ==========================================
def run_simulation(tester, mode, duration, scheduler):
    print(f"[*] Simulation Start | Mode: {mode.upper()} | Trajectory: Square")

    tester.mode = mode
    # 使用 Square 轨迹
    planner = TrajectoryPlanner("Rose", center_z=0.52, radius=0.15, period=duration)

    dt = tester.mj_model.opt.timestep
    steps = int(duration / dt)

    # --- Reset & Warmup ---
    tester.reset()
    start_pos = planner.get_target(0.0)
    tester.set_target(start_pos)
    tester.set_load([0, 0, 0])

    # 静态稳定
    for _ in range(500):
        tester.step()

    records = {
        "time": [],
        "ref": [],
        "act": [],
        "error": [],
        "force_mag": [],  # 记录力的大小用于验证
        "cmd_pcc": [],
        "cmd_rl": [],
        "cmd_total": [],
    }

    for i in range(steps):
        t = i * dt

        # 1. 获取当前时刻的复杂负载
        current_force = scheduler.get_force(t)
        tester.set_load(current_force)

        # 2. 获取目标
        target = planner.get_target(t)
        tester.set_target(target)

        # 3. 步进
        info = tester.step()

        # 4. 记录数据 (降采样)
        if i % 5 == 0:
            actual_pos = info["current_pos"]
            err = np.linalg.norm(target - actual_pos)

            records["time"].append(t)
            records["ref"].append(target)
            records["act"].append(actual_pos)
            records["error"].append(err)
            records["force_mag"].append(np.linalg.norm(current_force))
            records["cmd_pcc"].append(info["cmd_pcc"].copy())
            records["cmd_rl"].append(info["cmd_rl"].copy())
            records["cmd_total"].append(info["cmd_total"].copy())

    return records


# ==========================================
# 4. 绘图与分析
# ==========================================
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ... (前面的 TrajectoryPlanner, DisturbanceScheduler, run_simulation 保持不变) ...


def plot_complex_analysis(
    data_pcc, data_hybrid, scheduler, save_path="complex_load_test.png"
):
    t = np.array(data_pcc["time"])
    ref = np.array(data_pcc["ref"])
    act_pcc = np.array(data_pcc["act"])
    err_pcc = np.array(data_pcc["error"])
    act_hybrid = np.array(data_hybrid["act"])
    err_hybrid = np.array(data_hybrid["error"])

    # 1. 设置画布 (宽, 高)
    # 把它设宽一点，方便右图横向展开
    fig = plt.figure(figsize=(15, 6))

    # 使用 GridSpec 分割：左边占 1 份，右边占 2 份宽
    # 这会给右图预留更多的横向空间
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2], wspace=0.15)

    # -----------------------------------------
    # 左图：轨迹 (保持正方形)
    # -----------------------------------------
    ax_traj = fig.add_subplot(gs[0])
    ax_traj.plot(ref[:, 0], ref[:, 1], "k--", label="Ref")
    ax_traj.plot(act_pcc[:, 0], act_pcc[:, 1], "b-.", label="PCC")
    ax_traj.plot(act_hybrid[:, 0], act_hybrid[:, 1], "r-", label="Hybrid")

    ax_traj.set_xlabel("X (m)")
    ax_traj.set_ylabel("Y (m)")
    ax_traj.grid(True, linestyle=":", alpha=0.6)
    ax_traj.legend(loc="upper right")

    # [关键] 强制左图为正方形 (物理形状)
    ax_traj.set_box_aspect(1)

    # -----------------------------------------
    # 右图：误差 (自定义长宽比)
    # -----------------------------------------
    ax_err = fig.add_subplot(gs[1])

    ax_err.plot(t, err_pcc, "b-.", alpha=0.6, label="PCC")
    ax_err.plot(t, err_hybrid, "r-", label="Hybrid")

    # 绘制扰动区间
    for idx, (start, end) in enumerate(scheduler.active_intervals):
        ax_err.axvspan(start, end, color="gray", alpha=0.15)

    ax_err.set_xlabel("Time (s)")
    ax_err.set_ylabel("Error (m)")
    ax_err.set_ylim(bottom=0)
    ax_err.grid(True, alpha=0.3)
    ax_err.legend()

    # =========================================================
    # [核心修改] 强制设置右图的长宽比例 (Height / Width)
    # =========================================================

    # 如果您想要【扁长的长方形】(横轴长，纵轴短，适合时间序列):
    # 设置为小于 1 的数。例如 0.5 表示高度是宽度的一半。
    ax_err.set_box_aspect(0.4)

    # 如果您想要【瘦高的长方形】(纵轴长，横轴短):
    # 设置为大于 1 的数。例如 1.5 表示高度是宽度的 1.5 倍。
    # ax_err.set_box_aspect(1.5)

    # =========================================================

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_control_analysis(data_pcc, data_hybrid, scheduler, save_path="tra_control_analysis.png"):
    time_axis = np.array(data_hybrid["time"])
    err_pcc = np.array(data_pcc["error"])
    err_rl = np.array(data_hybrid["error"])
    
    cmd_rl = np.array(data_hybrid["cmd_rl"])
    cmd_pcc = np.array(data_hybrid["cmd_pcc"])
    cmd_total = np.array(data_hybrid["cmd_total"])
    
    colors = plt.cm.get_cmap("tab10", 8)
    actuator_labels = [f"Act {i+1}" for i in range(8)]
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 18), sharex=True)
    
    # 1. Error Comparison
    ax1 = axes[0]
    ax1.plot(time_axis, err_pcc, color="gray", linestyle="--", linewidth=1.5, alpha=0.8, label="PCC Baseline")
    ax1.plot(time_axis, err_rl, color="#d62728", linewidth=2.0, label="Hybrid (RL)")
    ax1.set_title("1. Tracking Error Comparison: PCC vs Hybrid", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Error (m)", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # 绘制扰动区间
    for idx, (start, end) in enumerate(scheduler.active_intervals):
        ax1.axvspan(start, end, color="gray", alpha=0.15, label="Disturbance Active" if idx == 0 else "")
    
    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys(), loc="upper right", framealpha=0.9)

    # 2. RL Residual Output
    ax2 = axes[1]
    ax2.set_title("2. RL Residual Output (Hybrid Mode)", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Delta Length (m)", fontsize=12)
    for i in range(8):
        ax2.plot(time_axis, cmd_rl[:, i], color=colors(i), linewidth=1.5, alpha=0.9, label=actuator_labels[i])
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    # 3. PCC Base Command
    ax3 = axes[2]
    ax3.set_title("3. PCC Base Command (Model-Based)", fontsize=14, fontweight="bold")
    ax3.set_ylabel("Base Length (m)", fontsize=12)
    for i in range(8):
        ax3.plot(time_axis, cmd_pcc[:, i], color=colors(i), linewidth=1.5, alpha=0.9)
    ax3.autoscale(enable=True, axis="y", tight=False)
    ax3.grid(True, linestyle="--", alpha=0.5)
    
    # 4. Total Control Command
    ax4 = axes[3]
    ax4.set_title("4. Total Control Command (PCC + RL)", fontsize=14, fontweight="bold")
    ax4.set_ylabel("Total Length (m)", fontsize=12)
    ax4.set_xlabel("Time (s)", fontsize=14)
    for i in range(8):
        ax4.plot(time_axis, cmd_total[:, i], color=colors(i), linewidth=1.5, alpha=0.9, label=actuator_labels[i])
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=8, fancybox=True, shadow=True, fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Plot] Saved control analysis to {save_path}")
    
    try:
        import matplotlib
        if matplotlib.get_backend().lower() != "agg":
            plt.show()
    except:
        pass
    finally:
        plt.close(fig)


# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    XML_FILE = "./source/two_disks_uj.xml"
    MODEL_FILE = "./models/legacy_replica_v3/final_model.zip"

    DURATION = 20.0  # 总时长

    # --- 定义4次复杂的负载事件 ---
    # 每次事件：0.1秒的强力冲击 (Impulse) + 之后的持续负载 (Step)
    # 分别在不同方向施加力，测试鲁棒性
    disturbance_events = [
        # Event 1 (t=4s): +X方向冲击，随后轻微拖拽
        {
            "start_time": 4.0,
            "impulse_vec": [10.0, 0.0, 0.0],
            "impulse_dur": 0.15,
            "step_vec": [3.0, 0.0, 0.0],
            "step_dur": 2.0,
        },
        # Event 2 (t=8s): -Y方向冲击，随后向下重物负载
        {
            "start_time": 8.0,
            "impulse_vec": [0.0, -10.0, 0.0],
            "impulse_dur": 0.15,
            "step_vec": [0.0, -2.0, -5.0],
            "step_dur": 2.5,
        },
        # Event 3 (t=13s): +Y方向强力冲击 (模拟碰撞)，随后持续反向推力
        {
            "start_time": 13.0,
            "impulse_vec": [0.0, 12.0, 0.0],
            "impulse_dur": 0.10,
            "step_vec": [0.0, 5.0, 0.0],
            "step_dur": 1.5,
        },
        # Event 4 (t=17s): 侧向混合冲击
        {
            "start_time": 17.0,
            "impulse_vec": [-8.0, 10.0, -5.0],
            "impulse_dur": 0.15,
            "step_vec": [-2.0, 2.0, 0.0],
            "step_dur": 1.5,
        },
    ]

    scheduler = DisturbanceScheduler(disturbance_events)

    tester = SoftRobotTester(
        xml_path=XML_FILE, model_path=MODEL_FILE, mode="hybrid", render=True
    )  # 关掉render加速

    # 1. Hybrid 运行
    data_hybrid = run_simulation(tester, "hybrid", DURATION, scheduler)

    # 2. PCC 运行
    data_pcc = run_simulation(tester, "pcc", DURATION, scheduler)

    # 3. 绘图
    plot_complex_analysis(data_pcc, data_hybrid, scheduler, save_path="rose.png")
    plot_control_analysis(data_pcc, data_hybrid, scheduler, save_path="tra_control_analysis.png")
