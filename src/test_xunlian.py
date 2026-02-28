import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ================= 配置区域 =================

LOG_DIR = "./logs/legacy_replica_v2"

# 只保留这一个 Tag
TAGS_OF_INTEREST = ["custom/step_reward_mean"]

# 截取前 50%
X_FRACTION = 0.5

# 纵轴范围 (聚焦于 -8 到 3)
Y_LIMITS = (-8, 3)

# 输出文件名
OUTPUT_FILENAME = "fig_reward_training.pdf"  # 推荐用 pdf 或 eps 矢量图格式

# ===========================================


def set_publication_style():
    """
    设置符合学术发表标准的绘图风格 (IEEE/Nature-like)
    """
    # 使用 seaborn 的 paper 上下文，调整字体缩放
    sns.set_context("paper", font_scale=1.5)

    # 启用刻度线风格
    sns.set_style("ticks")

    # 强制修改 Matplotlib 参数以匹配论文要求
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],  # 设置为 Times New Roman
            "axes.labelsize": 16,  # 轴标签字号
            "axes.titlesize": 16,  # 标题字号
            "xtick.labelsize": 16,  # x轴刻度字号
            "ytick.labelsize": 16,  # y轴刻度字号
            "legend.fontsize": 16,  # 图例字号
            "lines.linewidth": 2.5,  # 线宽
            "axes.grid": True,  # 开启网格
            "grid.alpha": 0.3,  # 网格透明度 (淡一点)
            "grid.linestyle": "--",  # 网格虚线
        }
    )


def smooth_curve_robust(series, window=100, method="median"):
    if method == "median":
        return series.rolling(window=window, min_periods=1, center=True).median()
    return series


def extract_tensorboard_data(root_log_dir, tags_to_extract):
    # (保持原有的提取逻辑不变)
    data_records = []
    event_files = glob.glob(
        os.path.join(root_log_dir, "**/*.tfevents*"), recursive=True
    )
    if not event_files:
        return pd.DataFrame()

    print(f"Processing {len(event_files)} log files...")
    for file_path in event_files:
        run_name = os.path.dirname(file_path).split(os.sep)[-1]
        try:
            ea = EventAccumulator(file_path)
            ea.Reload()
            available_tags = ea.Tags()["scalars"]
            for tag in tags_to_extract:
                if tag in available_tags:
                    events = ea.Scalars(tag)
                    for e in events:
                        data_records.append(
                            {
                                "step": e.step,
                                "value": e.value,
                                "tag": tag,
                                "run": run_name,
                            }
                        )
        except Exception:
            pass
    return pd.DataFrame(data_records)


def format_axis(ax):
    """
    美化坐标轴：科学计数法、去除边框等
    """
    # 1. 去除右侧和上侧的边框 (Despine)
    sns.despine(ax=ax, offset=5, trim=False)

    # 2. X 轴使用科学计数法 (例如 1.5 x 10^7)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.xaxis.get_offset_text().set_fontsize(12)  # 调整 1e7 的字体大小


def plot_paper_quality(df, output_dir="paper_figures"):
    if df.empty:
        print("Data is empty.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # --- 数据截取 ---
    max_step = df["step"].max()
    cutoff_step = max_step * X_FRACTION
    df = df[df["step"] <= cutoff_step]

    set_publication_style()

    unique_tags = df["tag"].unique()

    for tag in unique_tags:
        tag_df = df[df["tag"] == tag].sort_values(by="step")

        # 创建画布，长宽比通常为 4:3 或 16:9
        fig, ax = plt.subplots(figsize=(8, 5))

        # 1. 绘制原始数据 (Raw Data)
        #    技巧：使用非常淡的灰色，且线宽很细，作为背景噪声展示
        sns.lineplot(
            data=tag_df,
            x="step",
            y="value",
            errorbar="sd",
            label="Raw Data",
            alpha=0.2,  # 极高透明度
            color="#7f7f7f",  # 中性灰
            linewidth=0.8,  # 极细线宽
            ax=ax,
            legend=False,  # 暂时不加图例，后面统一加
        )

        # 2. 绘制平滑曲线 (Smoothed Trend)
        smoothed_rows = []
        for run_id in tag_df["run"].unique():
            run_data = tag_df[tag_df["run"] == run_id].copy()
            run_data["smoothed"] = smooth_curve_robust(
                run_data["value"], window=150, method="median"
            )
            smoothed_rows.append(run_data)

        if smoothed_rows:
            smooth_df = pd.concat(smoothed_rows)
            # 使用深红色或深蓝色，突出显示
            # color='#C44E52' (Seaborn Deep Red)
            # color='#1F77B4' (Seaborn Deep Blue)
            sns.lineplot(
                data=smooth_df,
                x="step",
                y="smoothed",
                errorbar=None,
                linewidth=2.5,
                label="Smoothed (Robust)",
                color="#C44E52",
                ax=ax,
            )

        # --- 3. 细节修饰 ---
        if Y_LIMITS:
            ax.set_ylim(Y_LIMITS)

        format_axis(ax)

        # 设置更专业的标签
        ax.set_xlabel("Training Steps (Timesteps)")
        # 将 custom/step_reward_mean 替换为论文用语
        if "reward" in tag:
            ax.set_ylabel("Average Step Reward")
        else:
            ax.set_ylabel("Value")

        # 标题通常论文里不需要（因为有 Figure Caption），如果需要可以留空或写简短
        # ax.set_title("Training Convergence", fontweight='bold')

        # 图例美化：去掉边框，放在合适的位置
        ax.legend(frameon=False, loc="lower right")

        plt.tight_layout()

        # 保存为 PDF (矢量图，论文首选) 和 PNG (预览用)
        base_name = tag.replace("/", "_")
        save_path_pdf = os.path.join(output_dir, f"{base_name}_paper.pdf")
        save_path_png = os.path.join(output_dir, f"{base_name}_paper.png")

        plt.savefig(save_path_pdf, dpi=300, bbox_inches="tight")
        plt.savefig(save_path_png, dpi=300, bbox_inches="tight")

        print(f"Saved PDF to: {save_path_pdf}")
        print(f"Saved PNG to: {save_path_png}")
        plt.close()


if __name__ == "__main__":
    df_data = extract_tensorboard_data(LOG_DIR, TAGS_OF_INTEREST)
    plot_paper_quality(df_data)
