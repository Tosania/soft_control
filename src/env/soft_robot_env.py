import os
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
from typing import Tuple, Dict, Optional
from src.core.control import SoftRobotModel, PCCController


class SimpleDisturbanceEnv(gym.Env):
    """
    极简版环境 (V8 - Minimalist Residual with velocity sensing):
    1. Obs 简化为 14 维 (误差向量 + 末端速度 + PCC基准)。
    2. 采用相对坐标，加速收敛。
    3. 奖励函数优化，鼓励顺从。
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 500}
    DEFAULT_CONFIG = {
        # 奖励函数的权重系数
        "reward_weights": {
            "dist": 2.0,  # 距离奖励
            "smooth": 0.5,  # 动作平滑惩罚
            "effort": 0.5,  # 动作稀疏/能耗惩罚
        },
        # 奖励函数的计算超参
        "reward_scales": {
            "dist_scale": 5.0,  # 控制 tanh 的收敛陡峭程度
        },
        # 终止条件
        "limits": {
            "max_time": 20.0,  # 最大时长 (秒)
            "max_dist_error": 0.3,  # 最大允许偏离 (米)
            "strict_fail_penalty": 2.0,  # 触发终止时的额外惩罚
        },
    }

    def __init__(
        self,
        xml_path: str,
        env_config: Optional[Dict] = None,
        render_mode: Optional[str] = None,
    ):
        self.xml_path = xml_path
        self.render_mode = render_mode
        self.curriculum_progress = 0.0
        self.config = self.DEFAULT_CONFIG.copy()
        if env_config is not None:
            # 简单的嵌套字典更新逻辑
            for key, val in env_config.items():
                if isinstance(val, dict) and key in self.config:
                    self.config[key] = self.config[key].copy()  # 浅拷贝防止污染默认值
                    self.config[key].update(val)
                else:
                    self.config[key] = val
        # 状态缓存
        self.last_action = np.zeros(8)
        self.last_tip_pos = np.zeros(3)
        # --- 1. 物理参数 ---
        self.fixed_params = {
            "bend_stiffness": 5e8,
            "twist_stiffness": 1e11,
            "joint_damping": 30.0,
            "actuator_kp": 800.0,
        }

        # --- 2. 加载模型 ---
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file not found: {xml_path}")
        with open(xml_path, "r") as f:
            self.xml_template = f.read()
        self._load_model()

        # --- 3. 动作与观测 ---
        # 动作: Residual [-1, 1], 实际缩放由 residual_scale 控制
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

        # [关键] 观测改为 14 维 (增加末端速度感知)
        # 0-2: 末端位置误差 (Tip Error) [相对量]
        # 3-5: 末端速度 (Tip Velocity) [动态感知抗扰]
        # 6-13: PCC 基准控制量 (l_base) [基准意图]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )

        # 调参重点：残差缩放
        # 建议设小一点 (0.05 ~ 0.1)，保证 RL 只是微调，不会导致物理崩溃
        self.residual_scale = 0.08

        # --- 4. 控制器初始化 ---
        self.robot_math_model = SoftRobotModel(
            L_list=[0.5, 0.5], r_disk=0.08, base_pos=np.array([0, 0, 0.3])
        )

        # 状态变量
        self.current_target = np.zeros(3)
        self.disturbance_timer = 0
        self.current_disturbance = np.zeros(3)
        self.disturbance_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "ring_10_body"
        )

    def update_curriculum(self, progress: float):
        """
        更新当前训练进度
        progress: 0.0 ~ 1.0
        """
        self.curriculum_progress = np.clip(progress, 0.0, 1.0)

    def _load_model(self):
        xml_content = self.xml_template.format(
            bend_val=f"{self.fixed_params['bend_stiffness']:.1e}",
            twist_val=f"{self.fixed_params['twist_stiffness']:.1e}",
            damping_val=f"{self.fixed_params['joint_damping']:.1f}",
            kp_val=f"{self.fixed_params['actuator_kp']:.1f}",
        )
        self.model = mujoco.MjModel.from_xml_string(xml_content)
        self.data = mujoco.MjData(self.model)

        act_names = [
            "act_tendon_north_1",
            "act_tendon_south_1",
            "act_tendon_east_1",
            "act_tendon_west_1",
            "act_tendon_first",
            "act_tendon_second",
            "act_tendon_third",
            "act_tendon_fourth",
        ]
        self.act_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in act_names
        ]
        self.viewer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.last_action = np.zeros(8)
        self.last_tip_pos = self._get_tip_pos()
        # 重置控制器
        self.pcc_controller = PCCController(
            self.robot_math_model, dt=self.model.opt.timestep
        )
        self.current_target = self._sample_target()

        # 重置扰动
        self.disturbance_timer = 0
        self.current_disturbance = np.zeros(3)
        self.data.xfrc_applied[:, :] = 0

        # 预热：让 PCC 先跑 100 步，到达初始稳态
        current_l_base = np.zeros(8)
        max_warmup_steps = 300
        target_tolerance = 0.02  # 容忍度：1.5cm 以内视为预热完成
        warmup_prob = 1.0 - 0.9 * self.curriculum_progress

        # 掷骰子决定是否预热
        do_warmup = self.np_random.random() < warmup_prob

        # 如果需要预热，预热多少步？
        # 初期：预热步数多 (300步稳稳的)
        # 后期：预热步数少 (即使预热也只给一点点时间)
        if do_warmup:
            # 基础步数 50，额外步数随难度降低
            max_warmup_steps = int(100 * (1.0 - self.curriculum_progress))

            # 执行预热循环 (原代码逻辑)
            target_tolerance = 0.02
            for i in range(max_warmup_steps):
                tip_pos = self._get_tip_pos()
                dist = np.linalg.norm(tip_pos - self.current_target)
                l_base, _ = self.pcc_controller.step(self.current_target, tip_pos)
                self.data.ctrl[self.act_ids] = l_base
                mujoco.mj_step(self.model, self.data)
                # 早停
                if dist < target_tolerance and i > 50:
                    break
            
            # 将预热最后的结果赋值给 current_l_base 以便构建正确的初始观测
            current_l_base = l_base.copy()
        else:
            # Hard Mode: 不预热！
            # 计算一下初始 PCC 指令但不执行多步迭代，让 RL 面对初始的大误差
            tip_pos = self._get_tip_pos()
            current_l_base, _ = self.pcc_controller.step(self.current_target, tip_pos)

        # 预热完毕后，更新状态，防止第一步 step 算出极大的速度惩罚
        self.last_tip_pos = self._get_tip_pos()
        self.last_action = np.zeros(8)
        self.current_tip_velocity = np.zeros(3)

        # 获取初始观测时，把对应的 l_base 传进去，同时也会更新 mid_shape_error 
        obs = self._get_obs(l_base=current_l_base)

        return obs, {}

    def step(self, action):
        # 1. 施加扰动
        self._apply_random_disturbance()

        # 2. 获取当前状态用于 PCC 计算
        current_tip_pos = self._get_tip_pos()

        # 3. PCC 计算基准 (Teacher Policy)
        # 注意：PCC 应该总是基于“当前真实位置”来计算下一步
        l_base, _ = self.pcc_controller.step(self.current_target, current_tip_pos)

        # 4. RL 叠加残差
        # action [-1, 1] -> [-0.08, 0.08] (米)
        l_residual = action * self.residual_scale
        l_final = l_base + l_residual

        # 5. 物理限位
        # l_final[:4] = np.clip(l_final[:4], 0.1, 0.9)
        # l_final[4:] = np.clip(l_final[4:], 0.3, 1.7)

        # 6. 执行
        self.data.ctrl[self.act_ids] = l_final
        mujoco.mj_step(self.model, self.data)

        # 7. 计算奖励与状态更新
        new_tip_pos = self._get_tip_pos()
        self.current_tip_velocity = (new_tip_pos - self.last_tip_pos) / self.model.opt.timestep

        dist_vec = self.current_target - new_tip_pos
        distance = np.linalg.norm(dist_vec)

        # --- Reward Engineering ---
        # === [修改] 提取配置参数 ===
        weights = self.config["reward_weights"]
        scales = self.config["reward_scales"]
        limits = self.config["limits"]

        # A. 追踪奖励
        # 使用配置中的 dist_scale
        r_dist = 1.0 - (scales["dist_scale"] * distance)


        # C. 平滑与稀疏 (强调平滑补偿，允许合理发力)
        action_diff = action - self.last_action
        r_smooth = -np.linalg.norm(action_diff)
        
        # [修改] 听取您的建议：取消随距离放大的惩罚，允许远距离时 RL 输出必要的补偿量。
        # 这里仅做基础的稀疏性约束，防止无意义的乱动
        r_effort = -np.linalg.norm(action)



        # === 加权求和 ===
        reward = (
            weights["dist"] * r_dist
            + weights["effort"] * r_effort
            + weights["smooth"] * r_smooth
        )

        self.last_action = action.copy()
        self.last_tip_pos = new_tip_pos.copy()
        # 8. 终止条件
        terminated = False
        truncated = False
        if self.data.time > limits["max_time"]:
            truncated = True

        # 使用配置中的 max_dist_error
        if distance > limits["max_dist_error"]:
            terminated = True
            reward -= limits["strict_fail_penalty"]  # 额外的失败惩罚

        # 可视化 Target
        self.data.mocap_pos[0] = self.current_target
        if self.render_mode == "human":
            self._render_frame()

        obs = self._get_obs(l_base)

        # 记录每一个 reward component 对应的实际贡献 (乘过权重) 进 info
        info = {
            "dist": distance, 
            "l_base": l_base,
            "r_dist_weighted": weights["dist"] * r_dist,
            "r_effort_weighted": weights["effort"] * r_effort,
            "r_smooth_weighted": weights["smooth"] * r_smooth
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self, l_base):
        """
        极简 Obs (14维)，引入速度感知
        """
        real_tip_pos = self._get_tip_pos()

        # 1. 位置误差 (3) - 相对量，最关键
        tip_error = self.current_target - real_tip_pos
        
        # 2. 末端速度 (3) - 解决无法感知被外部推动的问题
        # 如果是 reset 刚进来，为 0
        if not hasattr(self, 'current_tip_velocity'):
            self.current_tip_velocity = np.zeros(3)
        tip_velocity = self.current_tip_velocity.copy()

        # 3. PCC 基准 (8) - 告知 RL 当前的大方向
        # 不需要给 current_l，因为 current_l ≈ l_base + last_action
        # RL 网络通常能隐式推断出这个关系

        obs = np.concatenate(
            [tip_error, tip_velocity, l_base]  # (3) + (3) + (8) = 14 维
        ).astype(np.float32)

        return obs

    def _sample_target(self):
        # 稍微缩小范围，让训练更容易起步
        theta = self.np_random.uniform(0, 2 * np.pi)
        r = self.np_random.uniform(0.2, 0.6)
        z = self.np_random.uniform(0.5, 0.9)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.array([x, y, z])

    def _apply_random_disturbance(self):
        """
        扰动逻辑 V3 (强制冷却版):
        状态机: [扰动中] -> (时间到) -> [强制休息] -> (时间到) -> [随机判定]
        """
        # 1. 如果计时器还没走完，保持现状
        if self.disturbance_timer > 0:
            self.disturbance_timer -= 1
            self.data.xfrc_applied[:, :] = 0
            if self.disturbance_body_id != -1:
                self.data.xfrc_applied[self.disturbance_body_id][
                    :3
                ] = self.current_disturbance
            return

        # 2. 计时器归零了，检查刚才是在干什么

        # --- 情况 A: 刚才在被扰动 (Current Force != 0) ---
        # 必须强制进入休息，给系统喘息时间
        if np.linalg.norm(self.current_disturbance) > 1e-6:
            self.current_disturbance = np.zeros(3)
            # 强制休息 100 - 200 步 (0.2s - 0.4s)，让弹簧阻尼系统稳下来
            self.disturbance_timer = self.np_random.integers(100, 200)
            # print("Disturbance END -> Cooling down")

        # --- 情况 B: 刚才在休息 (Current Force == 0) ---
        # 现在可以掷骰子决定是否开启新一轮折磨
        else:
            rand_val = self.np_random.random()

            # 2% 概率：脉冲 (Impulse)
            if rand_val < 0.02:
                # 力度适中，不要太大
                mag = self.np_random.uniform(5, 10)
                angle = self.np_random.uniform(0, 2 * np.pi)
                force = np.array([mag * np.cos(angle), mag * np.sin(angle), 0.0])

                self.current_disturbance = force
                # 持续极短：5-8步
                self.disturbance_timer = self.np_random.integers(20, 50)
                # print(f"Disturbance START: Impulse {mag:.1f}N")

            # 15% 概率：阶跃 (Step)
            elif rand_val < 0.17:
                # 力度较小，模拟持续干扰
                mag = self.np_random.uniform(0.1, 1.5)
                angle = self.np_random.uniform(0, 2 * np.pi)
                force = np.array([mag * np.cos(angle), mag * np.sin(angle), 0.0])

                self.current_disturbance = force
                # 持续较长：1s - 2s
                self.disturbance_timer = self.np_random.integers(500, 2000)
                # print(f"Disturbance START: Step {mag:.1f}N")

            # 剩下概率：继续休息 (Long Rest)
            else:
                self.current_disturbance = np.zeros(3)
                # 再多休息一会儿
                self.disturbance_timer = self.np_random.integers(50, 150)

        # 3. 应用当前的决定
        self.data.xfrc_applied[:, :] = 0
        if self.disturbance_body_id != -1:
            self.data.xfrc_applied[self.disturbance_body_id][
                :3
            ] = self.current_disturbance

    def _get_tip_pos(self):
        try:
            return self.data.site("rod_tip").xpos.copy()
        except:
            return self.data.body("ring_10_body").xpos.copy()

    def _render_frame(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
