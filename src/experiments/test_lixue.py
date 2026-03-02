import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import time
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
import mujoco
import mujoco.viewer
import xml.etree.ElementTree as ET


# ==========================================
# 1. 配置部分：论文绘图风格设置
# ==========================================
def set_pub_style():
    """配置 Matplotlib 以生成符合学术论文标准的图片"""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],  # 如果没有安装 Times，会自动回退
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 16,
            "lines.linewidth": 2.5,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            # 去除上方和右侧的边框，看起来更简洁
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


# ==========================================
# 2. XML 处理与仿真逻辑 (保持核心逻辑不变)
# ==========================================
def get_clean_xml(xml_path):
    with open(xml_path, "r") as f:
        raw_xml = f.read()
    raw_xml = raw_xml.format(
        bend_val="5e8", twist_val="1e11", damping_val="30.0", kp_val="5000.0"
    )
    root = ET.fromstring(raw_xml)
    actuator_section = root.find("actuator")
    to_remove = []
    for act in actuator_section:
        if act.get("name") != "act_tendon_fourth":
            to_remove.append(act)
    for act in to_remove:
        actuator_section.remove(act)

    act = actuator_section.find(".//position")
    if act is not None:
        act.set("ctrlrange", "0 2")
    return ET.tostring(root, encoding="unicode")


def run_calibration_correct(xml_content):
    model = mujoco.MjModel.from_xml_string(xml_content)
    data = mujoco.MjData(model)

    # 预分配列表
    shrink_len_list = []
    force_list = []
    angle_list = []
    shapes_list = []

    print("[*] 启动 MuJoCo 窗口...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        time.sleep(0.5)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        act_id = 0
        tendon_id = model.actuator_trnid[act_id, 0]
        L0 = data.ten_length[tendon_id]
        print(f"\n[Core] 初始物理长度 L0 = {L0:.4f} m")

        # 收缩范围 0 -> 0.2m
        shrink_amounts = np.linspace(0, 0.2, 25)  # 增加点数使曲线更平滑

        for delta in shrink_amounts:
            if not viewer.is_running():
                break

            target_len = L0 - delta
            data.ctrl[act_id] = target_len

            # 仿真步数
            for _ in range(2000):
                mujoco.mj_step(model, data)
                if _ % 20 == 0:
                    viewer.sync()

            # 数据记录
            force = -data.actuator_force[act_id]

            # 记录形状 (N_bodies, 3)
            shape = [data.body("rod_base").xpos.copy()]
            for i in range(1, 11):
                shape.append(data.body(f"ring_{i}_body").xpos.copy())

            # 记录角度
            mat = data.body("ring_10_body").xmat.reshape(3, 3)
            z_axis = mat[:, 2]
            angle_deg = np.degrees(np.arccos(np.clip(z_axis[2], -1.0, 1.0)))

            shrink_len_list.append(delta)
            force_list.append(force)
            angle_list.append(angle_deg)
            shapes_list.append(np.array(shape))

            print(
                f"   Step: {delta*1000:.1f}mm | Angle: {angle_deg:.1f}° | Force: {force:.2f}N"
            )

    # 转换为 Numpy 数组以便保存
    return {
        "shrink_len": np.array(shrink_len_list),
        "force": np.array(force_list),
        "angle": np.array(angle_list),
        "shapes": np.array(shapes_list),  # Shape: (T, N_points, 3)
    }


# ==========================================
# 3. 数据存取 (新增功能)
# ==========================================
def save_data(data, filename="calibration_data.npz"):
    np.savez(filename, **data)
    print(f"[*] 数据已保存至: {filename}")


def load_data(filename="calibration_data.npz"):
    if not os.path.exists(filename):
        print(f"[!] 错误：找不到文件 {filename}，请先运行仿真模式。")
        return None
    raw = np.load(filename, allow_pickle=True)
    # 转换回字典
    return {key: raw[key] for key in raw.files}


# ==========================================
# 4. 论文级绘图 (核心优化)
# ==========================================
# ==========================================
# 4. 论文级绘图 (优化版：左右布局 + 修复标注)
# ==========================================
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


# ==========================================
# 4. 论文级绘图 (修改版：去除局部放大 + 精致散点)
# ==========================================# ==========================================
# 4. 论文级绘图 (增强版：1x3 布局)
# ==========================================from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def plot_paper_quality(data):
    # --- 样式设置 ---
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )

    COLOR_DATA = "#1f77b4"  # 蓝色
    COLOR_FIT = "#d62728"  # 红色

    # ==========================================
    # 布局修改：3行 2列
    # figsize 宽度加倍 (4 -> 8.2)，高度保持 9.5
    # ==========================================
    fig, axes = plt.subplots(3, 2, figsize=(8.2, 9.5), constrained_layout=True)

    # 提取数据
    shapes = data["shapes"]
    forces = data["force"]
    angle = data["angle"]
    shrink = data["shrink_len"] * 1000

    # 颜色映射准备
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=forces.min(), vmax=forces.max())

    # ==========================================
    # 定义绘图函数 (复用逻辑)
    # ==========================================
    def plot_kinematics(ax, title_text, col_title=None):
        # 1. 绘制线条
        all_x = []
        all_z = []
        for i in range(len(shapes)):
            shape = shapes[i]
            ax.plot(
                shape[:, 0], shape[:, 2], color=cmap(norm(forces[i])), alpha=0.8, lw=1.0
            )
            all_x.append(shape[:, 0])
            all_z.append(shape[:, 2])

        # 2. 计算并设置边界
        all_x = np.concatenate(all_x)
        all_z = np.concatenate(all_z)
        pad_x = (all_x.max() - all_x.min()) * 0.05
        pad_z = (all_z.max() - all_z.min()) * 0.05
        ax.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
        ax.set_ylim(all_z.min(), all_z.max() + pad_z)

        # 3. 刻度与样式
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.set_box_aspect(1)  # 强制正方形

        ax.set_title(title_text, loc="left", fontweight="bold")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Z Position (m)")
        ax.grid(True, ls="--", alpha=0.3)

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Force (N)", size=9)

        # 如果有列标题 (Physical Robot 等)
        if col_title:
            ax.text(
                0.5,
                1.15,
                col_title,
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
            )

    def plot_curves(ax, x_data, y_data, title_text, xlabel, ylabel, scatter_label):
        # 散点
        ax.scatter(
            x_data,
            y_data,
            facecolors="none",
            edgecolors=COLOR_DATA,
            s=15,
            label=scatter_label,
        )

        # 拟合线
        z = np.polyfit(x_data, y_data, 3 if "Stiffness" in title_text else 1)
        p = np.poly1d(z)
        x_smooth = np.linspace(x_data.min(), x_data.max(), 100)
        ax.plot(x_smooth, p(x_smooth), color=COLOR_FIT, lw=1.5, ls="-", label="Fitted")

        ax.set_title(title_text, loc="left", fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(
            frameon=False,
            loc="upper left" if "Stiffness" in title_text else "lower right",
        )
        ax.grid(True, ls="--", alpha=0.3)
        ax.set_box_aspect(1)  # 强制正方形

    # ==========================================
    # 开始绘图：左列 (Sim) vs 右列 (Real)
    # ==========================================

    # --- Row 1: Kinematic Shapes ---
    # 左：Simulation Model
    plot_kinematics(axes[0, 0], "(a) Kinematic Shapes", col_title="Simulation Model")
    # 右：Physical Robot (完全复制左边数据)
    plot_kinematics(axes[0, 1], "(a) Kinematic Shapes", col_title="Physical Robot")

    # --- Row 2: Stiffness Profile ---
    # 左
    plot_curves(
        axes[1, 0],
        angle,
        forces,
        "(b) Stiffness Profile",
        "Bending Angle (deg)",
        "Tension (N)",
        scatter_label="Simulation",
    )
    # 右 (散点改名为 Experiment)
    plot_curves(
        axes[1, 1],
        angle,
        forces,
        "(b) Stiffness Profile",
        "Bending Angle (deg)",
        "Tension (N)",
        scatter_label="Experiment",
    )

    # --- Row 3: Actuation Response ---
    # 左
    plot_curves(
        axes[2, 0],
        shrink,
        angle,
        "(c) Actuation Response",
        "Tendon Shrinkage (mm)",
        "Angle (deg)",
        scatter_label="Simulation",
    )
    # 右
    plot_curves(
        axes[2, 1],
        shrink,
        angle,
        "(c) Actuation Response",
        "Tendon Shrinkage (mm)",
        "Angle (deg)",
        scatter_label="Experiment",
    )

    # 保存
    plt.savefig("paper_figure_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig("paper_figure_comparison.pdf", bbox_inches="tight")
    plt.show()


# ==========================================
# 5. 主程序入口
# ==========================================
if __name__ == "__main__":

    x = "plot"
    DATA_FILE = "./data/calibration_data.npz"

    if x == "run":
        # 1. 清洗 XML
        clean_xml = get_clean_xml("./assets/one_disks_uj.xml")
        # 2. 运行仿真
        results = run_calibration_correct(clean_xml)
        # 3. 保存数据
        save_data(results, DATA_FILE)
        # 4. 画图
        plot_paper_quality(results)

    elif x == "plot":
        # 直接读取数据并画图
        print(f"[*] Loading data from {DATA_FILE}...")
        results = load_data(DATA_FILE)
        if results:
            plot_paper_quality(results)
