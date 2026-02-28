import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tester import SoftRobotTester
import matplotlib
import concurrent.futures
import multiprocessing

matplotlib.use("Agg")
sns.set_theme(style="white", context="paper")

# --- 全局变量 (用于多进程中的 Worker) ---
worker_tester = None


def init_worker():
    """多进程 Worker 初始化函数"""
    global worker_tester
    worker_tester = None


def get_process_tester(xml_path, model_path):
    """获取当前进程的 Tester 实例"""
    global worker_tester
    if worker_tester is None:
        worker_tester = SoftRobotTester(
            xml_path=xml_path, model_path=model_path, mode="hybrid", render=False
        )
    return worker_tester


def evaluate_single_point(args):
    """
    进程工作函数：评估单个空间点
    (修改版：记录全过程误差)
    """
    x, z, xml_path, model_path, force_vec, max_steps = args

    tester = get_process_tester(xml_path, model_path)
    target = np.array([x, 0.0, z])

    def run_test_mode(mode_name):
        tester.mode = mode_name
        tester.reset()
        tester.set_target(target)
        tester.set_load(force_vec)

        errors = []
        last_pos = None
        stable_count = 0

        # === 单一阶段循环：从 t=0 开始记录 ===
        for i in range(max_steps):
            info = tester.step()
            curr_pos = info["current_pos"]
            curr_error = info["error"]

            # 1. 每一发都记录误差
            errors.append(curr_error)

            # 2. 检查稳定性 (早停机制)
            # 虽然我们要记录全过程，但如果已经完全静止了，后面几千步全是重复数据，
            # 为了计算效率，我们可以认为过程已结束。
            if last_pos is not None:
                move_dist = np.linalg.norm(curr_pos - last_pos)
                if move_dist < 1e-5:  # 极小移动阈值
                    stable_count += 1
                else:
                    stable_count = 0
            last_pos = curr_pos

            # 连续 30 步位置不变，且至少跑了 50 步(给一点启动时间)，则提前结束
            if stable_count > 30 and i > 50:
                break

        # 返回全过程的平均误差 (Mean Absolute Error of Trajectory)
        return np.mean(errors)

    # 分别测试
    err_pcc = run_test_mode("pcc")
    err_rl = run_test_mode("hybrid")

    return (x, z, err_pcc, err_rl)


class WorkspaceHeatmapEvaluatorFast:
    def __init__(self, xml_path, model_path=None):
        self.xml_path = xml_path
        self.model_path = model_path

    def run_scan_multiprocess(
        self,
        x_range=[-0.4, 0.4],
        z_range=[0.5, 1.3],
        resolution=20,
        force_vec=[0, 0, -1.0],
        max_steps=2000,  # 修改参数名：最大步数
        max_workers=None,
    ):
        if max_workers is None:
            max_workers = multiprocessing.cpu_count() - 2
            if max_workers < 1:
                max_workers = 1

        print(f"[*] Starting Multi-PROCESS Scan ({resolution}x{resolution} points)...")
        print(f"[*] Strategy: Full Trajectory Recording (Max {max_steps} steps)")

        # 1. 生成网格
        xs = np.linspace(x_range[0], x_range[1], resolution)
        zs = np.linspace(z_range[0], z_range[1], resolution)
        X, Z = np.meshgrid(xs, zs)

        # 2. 生成任务
        tasks = []
        for i in range(resolution):
            for j in range(resolution):
                tasks.append(
                    (
                        X[i, j],
                        Z[i, j],
                        self.xml_path,
                        self.model_path,
                        force_vec,
                        max_steps,  # 传入新的参数
                    )
                )

        # 3. 并行执行
        chunk_size = max(1, len(tasks) // (max_workers * 4))
        start_t = time.time()

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers, initializer=init_worker
        ) as executor:
            results = list(
                tqdm(
                    executor.map(evaluate_single_point, tasks, chunksize=chunk_size),
                    total=len(tasks),
                    desc="Scanning Trajectories",
                )
            )

        # 4. 填充数据
        self.X, self.Z = X, Z
        self.err_pcc = np.zeros_like(X)
        self.err_rl = np.zeros_like(X)

        idx = 0
        for i in range(resolution):
            for j in range(resolution):
                _, _, e_pcc, e_rl = results[idx]
                self.err_pcc[i, j] = e_pcc
                self.err_rl[i, j] = e_rl
                idx += 1

        elapsed = time.time() - start_t
        print(f"[*] Complete in {elapsed:.2f}s")

        self.improvement = self.err_pcc - self.err_rl
        self.scan_config = {"force": force_vec}

    def plot_results(self, save_path="workspace_heatmap_trajectory.png"):
        """绘图逻辑不变"""
        if not hasattr(self, "X"):
            print("Please run scan() first.")
            return

        fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
        # 这里的 vmax 可能需要调整，因为包含瞬态误差，数值可能会比纯稳态大一些
        v_max_err = np.percentile(self.err_pcc, 98)

        # PCC
        im1 = axes[0].pcolormesh(
            self.X,
            self.Z,
            self.err_pcc,
            cmap="magma_r",
            shading="auto",
            vmin=0,
            vmax=v_max_err,
        )
        axes[0].set_title(
            f"PCC Full-Traj Error (m)\nForce={self.scan_config['force']}N", fontsize=14
        )
        axes[0].set_aspect("equal")
        fig.colorbar(im1, ax=axes[0])

        # RL
        im2 = axes[1].pcolormesh(
            self.X,
            self.Z,
            self.err_rl,
            cmap="magma_r",
            shading="auto",
            vmin=0,
            vmax=v_max_err,
        )
        axes[1].set_title("RL Hybrid Full-Traj Error (m)", fontsize=14)
        axes[1].set_aspect("equal")
        fig.colorbar(im2, ax=axes[1])

        # Improvement
        limit = max(abs(self.improvement.min()), abs(self.improvement.max()))
        im3 = axes[2].pcolormesh(
            self.X,
            self.Z,
            self.improvement,
            cmap="RdBu_r",
            shading="auto",
            vmin=-limit,
            vmax=limit,
        )
        axes[2].set_title("Improvement (Blue=Better)", fontsize=14)
        axes[2].set_aspect("equal")
        fig.colorbar(im3, ax=axes[2])

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"[*] Saved to {save_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()

    XML_FILE = "./source/two_disks_uj.xml"
    MODEL_FILE = "./models/legacy_replica_v1/final_model.zip"

    evaluator = WorkspaceHeatmapEvaluatorFast(XML_FILE, MODEL_FILE)

    evaluator.run_scan_multiprocess(
        x_range=[-0.5, 0.5],
        z_range=[0.5, 1.3],
        resolution=5,  # 保持高分辨率
        force_vec=[1, 0, -2.0],  # 建议测试一个中等负载
        max_steps=4000,  # 最大步数，足够到达稳态即可
        max_workers=25,  # 你的核心数
    )

    evaluator.plot_results(save_path="workspace_heatmap_full_process.png")
