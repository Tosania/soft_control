import os
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np  # <--- 记得导入 numpy
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    ProgressBarCallback,
    BaseCallback,
)  # <--- 导入 BaseCallback
from typing import Dict, Any, Optional

# 引用你的环境
from src.env.soft_robot_env import SimpleDisturbanceEnv


class CurriculumCallback(BaseCallback):
    """
    课程学习回调函数：
    根据训练总步数，线性增加环境难度。
    """

    def __init__(self, total_timesteps: int, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        # 1. 计算当前进度 (0.0 ~ 1.0)
        current_progress = self.num_timesteps / self.total_timesteps
        self.logger.record("curriculum/difficulty_progress", current_progress)

        # 如果你想看具体的“预热概率”，也可以算出来记进去，直观一点：
        # (假设 env 里是 1.0 - 0.9 * progress)
        warmup_prob = 1.0 - 0.9 * current_progress
        self.logger.record("curriculum/warmup_probability", warmup_prob)
        # 2. 为了减少通信开销，每 1000 步更新一次环境即可，不用每一步都更
        if self.num_timesteps % 1000 == 0:
            # env_method 可以调用 SubprocVecEnv 中所有子环境的方法
            # 这行代码等价于：对每个 env 执行 env.update_curriculum(current_progress)
            self.training_env.env_method("update_curriculum", current_progress)

            if self.verbose > 0 and self.num_timesteps % 50000 == 0:
                print(f"[Curriculum] Difficulty updated to {current_progress:.2f}")

        return True


# ==========================================
# 1. 移植自定义回调函数 (用于记录物理指标)
# ==========================================
class TensorboardCallback(BaseCallback):
    """
    自定义回调：将物理指标 'dist' (追踪误差) 和瞬时奖励记录到 TensorBoard
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # self.locals['infos'] 包含所有并行环境当前步的 info 字典
        infos = self.locals.get("infos", [])

        if infos:
            # 提取所有环境的 'dist' (我们在 Env 的 step 函数 info 里放入了 'dist')
            dists = [info.get("dist") for info in infos if "dist" in info]
            if dists:
                mean_dist = np.mean(dists)
                # 记录核心指标：平均追踪误差 (m)
                self.logger.record("custom/tracking_error", mean_dist)

            for val_key in ["r_dist_weighted", "r_effort_weighted", "r_smooth_weighted"]:
                vals = [info.get(val_key) for info in infos if val_key in info and info.get(val_key) is not None]
                if vals:
                    self.logger.record(f"reward_components/{val_key}", np.mean(vals))

        # 记录瞬时奖励 (Step Reward) 用于分析奖励函数的稠密程度
        rewards = self.locals.get("rewards", [])
        if rewards is not None and len(rewards) > 0:
            mean_reward = np.mean(rewards)
            self.logger.record("custom/step_reward_mean", mean_reward)

        return True


# ==========================================
# 2. 修正后的 Trainer 类
# ==========================================
class SoftRobotTrainer:
    def __init__(
        self,
        experiment_name: str,
        base_log_dir: str = "./logs/training",
        base_model_dir: str = "./checkpoints",
    ):
        self.exp_name = experiment_name
        self.log_dir = os.path.join(base_log_dir, experiment_name)
        self.model_dir = os.path.join(base_model_dir, experiment_name)

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    def run(
        self,
        xml_path: str,
        train_params: Optional[Dict] = None,
        env_config: Optional[Dict] = None,
    ):

        # 默认参数
        tp = {
            "total_timesteps": 1_000_000,
            "num_envs": 8,
            "learning_rate": 3e-4,
            "seed": 42,
            "n_steps": 128,
            "batch_size": 128,
            "net_arch": [128, 128],
        }
        if train_params:
            tp.update(train_params)

        print(f"\n{'='*60}")
        print(f"🚀 Starting Experiment: {self.exp_name}")
        print(f"📂 Log Dir: {self.log_dir}")
        print(f"⚙️  Env Config: {env_config if env_config else 'Default'}")
        print(f"{'='*60}\n")

        # 工厂函数
        def make_env(rank, seed):
            def _init():
                env = SimpleDisturbanceEnv(
                    xml_path=xml_path, env_config=env_config, render_mode=None
                )
                env.reset(seed=seed + rank)
                return env

            return _init

        # 创建环境
        # 注意：如果在 Windows 上遇到多进程报错，请改用 DummyVecEnv
        env = SubprocVecEnv([make_env(i, tp["seed"]) for i in range(tp["num_envs"])])
        env = VecMonitor(env, self.log_dir)

        # 初始化模型
        policy_kwargs = dict(net_arch=tp["net_arch"])
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=tp["learning_rate"],
            n_steps=tp["n_steps"],
            batch_size=tp["batch_size"],
            tensorboard_log=self.log_dir,
            policy_kwargs=policy_kwargs,
            device="auto",
        )

        # === [关键修改] ===
        # 实例化自定义回调
        curriculum_callback = CurriculumCallback(
            total_timesteps=tp["total_timesteps"], verbose=1
        )
        tb_callback = TensorboardCallback()

        checkpoint_callback = CheckpointCallback(
            save_freq=50_000, save_path=self.model_dir, name_prefix="ckpt"
        )

        # 将 tb_callback 加入列表
        callbacks_list = [
            ProgressBarCallback(),
            checkpoint_callback,
            tb_callback,
            curriculum_callback,
        ]

        try:
            model.learn(
                total_timesteps=tp["total_timesteps"],
                callback=callbacks_list,  # <--- 传入列表
                tb_log_name="PPO",
            )
            model.save(f"{self.model_dir}/final_model")
            print(
                f"\n✅ Training Finished. Model saved to {self.model_dir}/final_model.zip"
            )

        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted. Saving current model...")
            model.save(f"{self.model_dir}/interrupted_model")

        finally:
            env.close()


# ... (main 部分保持不变) ...

# ==========================================
# 使用示例：如何运行实验
# ==========================================
if __name__ == "__main__":
    XML_FILE = "./assets/two_disks_uj.xml"
    trainer = SoftRobotTrainer(experiment_name="legacy_replica_v7")
    trainer.run(
        xml_path="./assets/two_disks_uj.xml",
        # 1. 训练超参数 (复刻 train.py)
        train_params={
            "total_timesteps": 2000_0000,
            "num_envs": 32,
            "seed": 1048,
            "learning_rate": 3e-4,
            "n_steps": 512,
            "batch_size": 4096,
            "net_arch": [128, 128],
        },
        # 2. 环境逻辑参数 (还原至 v3 状态)
        env_config={
            "reward_weights": {
                "dist": 2.0, 
                "smooth": 0.5,
                "effort": 0.5,
            },
            "reward_scales": {
                "dist_scale": 5.0, 
            },
            "limits": {
                "max_time": 20.0,  
                "max_dist_error": 0.3,
                "strict_fail_penalty": 2.0,
            },
        },
    )
    #os.system("shutdown -h now")
