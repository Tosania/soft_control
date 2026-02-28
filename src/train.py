import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, ProgressBarCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    """
    自定义回调函数：用于将 'dist' (追踪误差) 记录到 TensorBoard
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # self.locals['infos'] 是一个列表，包含所有并行环境当前步的 info 字典
        infos = self.locals.get("infos", [])

        if infos:
            # 提取所有环境的 'dist'
            # 注意：如果环境还没reset，第一步可能没有dist，给个默认值或者跳过
            dists = [info.get("dist") for info in infos if "dist" in info]

            if dists:
                # 计算平均误差
                mean_dist = np.mean(dists)
                # 记录到 TensorBoard，标签叫 "custom/tracking_error"
                self.logger.record("custom/tracking_error", mean_dist)
        rewards = self.locals.get("rewards", [])

        if rewards is not None and len(rewards) > 0:
            mean_reward = np.mean(rewards)
            # 记录平均瞬时奖励
            self.logger.record("custom/step_reward_mean", mean_reward)
        return True


# 导入你刚刚写好的简化环境
# 假设文件名叫 simple_env.py，类名叫 SimpleDisturbanceEnv
from soft_robot_env import SimpleDisturbanceEnv
from typing import Callable


# [新增] 线性学习率调度器
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        """
        return progress_remaining * initial_value

    return func


# ==========================================
# 1. 训练配置
# ==========================================
CONFIG = {
    "total_timesteps": 80_00000,  # 100万步，足够收敛
    "num_envs": 32,  # 并行环境数 (根据你CPU核数调整，例如 8 或 16)
    "seed": 1048,
    "log_dir": "./logs/simple_dist_test/",
    "model_dir": "./models/simple_dist_test/",
    "xml_path": "./source/two_disks_uj.xml",
}


def make_env(rank, seed):
    """
    环境工厂函数
    """

    def _init():
        # 初始化我们写的简化环境
        env = SimpleDisturbanceEnv(
            xml_path=CONFIG["xml_path"], render_mode=None  # 训练时不要渲染，速度快
        )

        # 可以在这里包装 TimeLimit，但我们在 Env 内部已经写了截断逻辑
        env.reset(seed=seed + rank)
        return env

    return _init


# ==========================================
# 2. 主程序
# ==========================================
if __name__ == "__main__":
    # 1. 目录准备
    os.makedirs(CONFIG["log_dir"], exist_ok=True)
    os.makedirs(CONFIG["model_dir"], exist_ok=True)

    print(f"=== 开始简化环境训练 (SimpleDisturbanceEnv) ===")
    print(f"Observation包含了PCC基准量，RL仅学习残差。")

    # 2. 创建向量化环境
    # SubprocVecEnv 利用多核 CPU 加速采样
    # 如果在 Windows 上报错，可以改用 DummyVecEnv (但速度会慢)
    env = SubprocVecEnv(
        [make_env(i, CONFIG["seed"]) for i in range(CONFIG["num_envs"])]
    )
    # VecMonitor 用于记录 Reward 曲线，方便 TensorBoard 查看
    env = VecMonitor(env, CONFIG["log_dir"])

    # 3. 定义模型 (Standard PPO)
    # 策略网络使用 MLP (多层感知机)
    # 考虑到我们输入了 PCC 参考量，网络不需要太深，[64, 64] 或 [128, 128] 即可
    policy_kwargs = dict(net_arch=[128, 128])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=512,  # 每个环境采样多少步更新一次
        batch_size=128 * 8,
        learning_rate=(3e-4),
        ent_coef=0.005,  # 熵系数，控制探索。如果过早收敛到局部最优，可设为 0.01
        policy_kwargs=policy_kwargs,
        tensorboard_log=CONFIG["log_dir"],
        device="auto",  # 自动检测 GPU
    )
    from stable_baselines3.common.callbacks import CallbackList

    # 1. 实例化我们的自定义 Callback
    tb_callback = TensorboardCallback()
    # 4. 回调函数
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000, save_path=CONFIG["model_dir"], name_prefix="simple_model"
    )

    # 5. 开始训练
    try:
        model.learn(
            total_timesteps=CONFIG["total_timesteps"],
            callback=[ProgressBarCallback(), checkpoint_callback, tb_callback],
            tb_log_name="PPO_Simple_Res",
        )

        # 保存最终模型
        model.save(f"{CONFIG['model_dir']}/final_model")
        print("训练完成！模型已保存。")

    except KeyboardInterrupt:
        print("\n训练被手动中断，正在保存当前模型...")
        model.save(f"{CONFIG['model_dir']}/interrupted_model")

    finally:
        env.close()
        print("环境已关闭。")
    # os.system("shutdown -h now")
    # ==========================================
    # 6. (可选) 简单的可视化测试
    # ==========================================
    # 只有当在本地运行且想看效果时才取消注释
    """
    print("正在加载模型进行可视化测试...")
    test_env = SimpleDisturbanceEnv(CONFIG["xml_path"], render_mode="human")
    model = PPO.load(f"{CONFIG['model_dir']}/final_model")
    
    obs, _ = test_env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        if done or truncated:
            obs, _ = test_env.reset()
    test_env.close()
    """
