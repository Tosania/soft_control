import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# ==========================================
# 1. 论文级样式配置 (Global Style)
# ==========================================
def set_academic_style():
    """
    配置 Matplotlib 以生成符合 Soft Robotics/IEEE 顶刊标准的图片
    """
    plt.rcParams.update(
        {
            # 字体设置
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",  # 类似 LaTeX 的数学字体
            "font.size": 10,
            # 坐标轴设置
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.linewidth": 1.0,  # 边框稍微加粗
            "axes.grid": True,
            "grid.color": "#EAEAEA",  # 极淡的网格
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            # 刻度设置 (向内，显得更专业)
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.top": True,  # 上方也有刻度
            "ytick.right": True,  # 右侧也有刻度
            # 图例
            "legend.frameon": False,  # 去掉图例边框
            "legend.fontsize": 9,
            # 图片输出
            "figure.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


# ==========================================
# 2. 辅助函数：强制正方形视口
# ==========================================
def make_axis_square_equal(ax, x_data, z_data, pad=0.1):
    """
    强制设置 Axis 的 Aspect 为 Equal (物理真实)，
    同时调整 Lim 使得最终的绘图区域是正方形 (Square Box)。
    """
    # 1. 获取数据的物理范围
    x_min, x_max = np.min(x_data), np.max(x_data)
    z_min, z_max = np.min(z_data), np.max(z_data)

    w = x_max - x_min
    h = z_max - z_min

    # 2. 找出最大的跨度，作为正方形的边长
    max_span = max(w, h) * (1.0 + pad)  # 增加一点余量

    # 3. 计算中心点
    x_center = (x_max + x_min) / 2
    z_center = (z_max + z_min) / 2

    # 4. 设置新的 Limit，使两个方向跨度一致
    ax.set_xlim(x_center - max_span / 2, x_center + max_span / 2)
    ax.set_ylim(z_center - max_span / 2, z_center + max_span / 2)

    # 5. 强制物理比例一致
    ax.set_aspect("equal", adjustable="box")


# ==========================================
# 3. 核心绘图函数
# ==========================================
def plot_paper_quality(data):
    # 应用样式
    set_academic_style()

    # 颜色定义 (参考 Science/Nature 常用色)
    # Deep Blue (数据), Vermilion (拟合/模型), Yellow-Green (Map)
    C_DATA = "#00468B"  # 深蓝
    C_FIT = "#ED0000"  # 鲜红
    C_GREY = "#333333"

    # 创建 3行1列 的布局 (符合提供的参考图垂直构图)
    # figsize=(width, height) 单位英寸
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(4, 10))

    # ==========================
    # (a) Kinematic Shapes
    # ==========================
    shapes = data["shapes"]  # Shape: (N, Points, 3)
    forces = data["force"]

    # 使用 Viridis 并且截取中间一段，避免太黄或太紫
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=forces.min(), vmax=forces.max())

    # 绘制每一根线条
    # 使用 LineCollection 优化性能并获得更好的抗锯齿效果
    from matplotlib.collections import LineCollection

    lines = []
    colors = []

    all_x = []
    all_z = []

    for i in range(len(shapes)):
        # 提取 X 和 Z (MuJoCo 中 Z 是向上)
        x = shapes[i][:, 0]
        z = shapes[i][:, 2]

        # 收集数据用于计算边界
        all_x.append(x)
        all_z.append(z)

        # 构建线段
        points = np.array([x, z]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # 绘图
        lc = LineCollection(
            segments, colors=cmap(norm(forces[i])), linewidths=1.2, alpha=0.8
        )
        ax1.add_collection(lc)

    # === 关键：强制正方形逻辑 ===
    flat_x = np.concatenate(all_x)
    flat_z = np.concatenate(all_z)
    make_axis_square_equal(ax1, flat_x, flat_z, pad=0.1)

    ax1.set_xlabel("X Position (m)")
    ax1.set_ylabel("Z Position (m)")
    # 标题左对齐，加粗
    ax1.set_title("(a) Kinematic Shapes", loc="left", fontweight="bold", pad=10)

    # --- 嵌入式 Colorbar (Inset) ---
    # 位置：右上角内部
    axins = inset_axes(
        ax1,
        width="5%",
        height="40%",
        loc="upper right",
        bbox_to_anchor=(-0.05, -0.05, 1, 1),  # 微调位置
        bbox_transform=ax1.transAxes,
    )
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=axins)
    cbar.ax.tick_params(
        labelsize=8, left=True, right=False, labelleft=True, labelright=False
    )
    cbar.set_label(
        "Force (N)", fontsize=9, labelpad=-35, y=0.5
    )  # Label 放在左侧防止出界

    # ==========================
    # (b) Stiffness Profile
    # ==========================
    angle = data["angle"]
    force = data["force"]

    # 散点：空心圆，看起来更精致
    ax2.scatter(
        angle,
        force,
        facecolors="none",
        edgecolors=C_DATA,
        s=20,
        linewidth=1.0,
        label="Sim.",
        zorder=2,
    )

    # 拟合：三次多项式
    z2 = np.polyfit(angle, force, 3)
    p2 = np.poly1d(z2)
    x_range = np.linspace(min(angle), max(angle), 100)
    ax2.plot(
        x_range, p2(x_range), color=C_FIT, lw=1.5, linestyle="-", label="Fit", zorder=1
    )

    ax2.set_xlabel("Bending Angle (deg)")
    ax2.set_ylabel("Actuation Tension (N)")
    ax2.set_title("(b) Stiffness Profile", loc="left", fontweight="bold", pad=10)

    # 强制正方形框 (虽然数据比例不equal，但框要是方的)
    ax2.set_box_aspect(1)
    # 图例放在左上角，无边框
    ax2.legend(loc="upper left")

    # ==========================
    # (c) Actuation Response
    # ==========================
    shrink = data["shrink_len"] * 1000  # m -> mm

    # 散点
    ax3.scatter(
        shrink,
        angle,
        facecolors="none",
        edgecolors="#2E7D32",
        s=20,
        linewidth=1.0,
        zorder=2,
    )

    # 线性拟合
    z3 = np.polyfit(shrink, angle, 1)
    p3 = np.poly1d(z3)
    x_range_3 = np.linspace(min(shrink), max(shrink), 100)

    ax3.plot(
        x_range_3,
        p3(x_range_3),
        color="#2E7D32",
        lw=1.5,
        linestyle="-",
        label="Model",
        zorder=1,
    )

    ax3.set_xlabel("Tendon Shrinkage (mm)")
    ax3.set_ylabel("Bending Angle (deg)")
    ax3.set_title("(c) Actuation Response", loc="left", fontweight="bold", pad=10)
    ax3.set_box_aspect(1)
    ax3.legend(loc="lower right")

    # 调整整体留白
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)  # 增加垂直间距，避免标签重叠

    # 保存
    plt.savefig("paper_figure_vertical.png", dpi=300)
    plt.savefig("paper_figure_vertical.pdf")  # 矢量图，适合插入 LaTeX
    plt.show()


# ==========================================
# 主程序调用逻辑 (保持不变)
# ==========================================
if __name__ == "__main__":
    # 假设你已经有了 calibration_data.npz
    # 直接加载绘图
    DATA_FILE = "calibration_data.npz"
    import os

    if os.path.exists(DATA_FILE):
        raw = np.load(DATA_FILE, allow_pickle=True)
        results = {key: raw[key] for key in raw.files}
        plot_paper_quality(results)
    else:
        print("请先运行仿真生成数据！")
