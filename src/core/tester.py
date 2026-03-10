import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import time
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from src.core.control import SoftRobotModel, PCCController

from collections import deque

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


class RealTimePlotter:
    """
    高级实时误差显示器
    特性：
    1. 保留所有历史数据（不删除旧数据）。
    2. 渐变色线条：颜色随误差大小动态变化（低误差绿 -> 高误差红）。
    3. 附带 Colorbar 对比条。
    """

    def __init__(self, window_title="Control Error Analysis", max_error_limit=0.5):
        # 初始化数据容器（不再使用 deque，改为普通 list 以保留无限历史）
        self.times = []
        self.errors = []

        # 开启交互模式
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(9, 5))  # 稍微宽一点放 colorbar
        self.fig.canvas.manager.set_window_title(window_title)

        # --- 1. 设置颜色映射 (Colormap) ---
        # 'RdYlGn_r' 代表 Red-Yellow-Green (Reversed)，即：低值绿，高值红
        # 你也可以换成 'jet', 'viridis', 'plasma' 等
        self.cmap = plt.get_cmap("RdYlGn_r")

        # 设置归一化范围 (0米是最好的绿色，max_error_limit是纯红)
        self.norm = Normalize(vmin=0, vmax=max_error_limit)

        # --- 2. 初始化 LineCollection ---
        # 我们一开始创建一个空的集合，后续在 update 中填充
        self.lc = LineCollection([], cmap=self.cmap, norm=self.norm, linewidths=2.0)
        self.ax.add_collection(self.lc)

        # --- 3. 添加侧边颜色条 (Colorbar) ---
        self.cbar = self.fig.colorbar(self.lc, ax=self.ax)
        self.cbar.set_label("Error Magnitude (m)")

        # --- 4. 坐标轴设置 ---
        self.ax.set_xlabel("Time Steps")
        self.ax.set_ylabel("Euclidean Error (m)")
        self.ax.set_title("Real-time Gradient Tracking Performance")
        self.ax.grid(True, linestyle="--", alpha=0.5)

        # 初始视窗范围
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, max_error_limit)

    def update(self, error):
        # 1. 存入新数据
        self.errors.append(error)
        step_idx = len(self.errors)
        self.times.append(step_idx)

        # 至少要有两个点才能画线段
        if len(self.errors) < 2:
            return

        # --- 2. 构建渐变线段 ---
        # 将数据转换为 numpy 数组以便进行矩阵操作
        t_arr = np.array(self.times)
        e_arr = np.array(self.errors)

        # 创建点集 (x, y)，形状为 (N, 2)
        points = np.array([t_arr, e_arr]).T.reshape(-1, 1, 2)

        # 创建线段集：连接点 i 和点 i+1
        # segments 形状为 (N-1, 2, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # --- 3. 更新绘图对象 ---
        self.lc.set_segments(segments)
        # 用误差值本身来决定颜色（这里用起点的误差代表该段颜色，也可取平均）
        self.lc.set_array(e_arr[:-1])

        # --- 4. 动态调整视窗 ---
        # X轴：始终跟随数据增长
        self.ax.set_xlim(0, step_idx + 10)

        # Y轴：根据最大误差自动调整，但保留一点头部空间
        current_max = e_arr.max()
        if current_max > self.ax.get_ylim()[1]:
            self.ax.set_ylim(0, current_max * 1.2)
            # 如果误差超过了最初设定的颜色上限，也可以动态调整颜色的归一化范围（可选）
            # self.lc.set_norm(Normalize(vmin=0, vmax=current_max))

        # --- 5. 刷新 ---
        # 这里的 pause 很关键，既能刷新界面，又防止界面卡死
        plt.pause(0.001)

    def close(self):
        plt.ioff()
        plt.close(self.fig)


class SoftRobotTester:
    def __init__(self, xml_path, model_path=None, mode="pcc", render=True, video=False, obs_type="tip_velocity"):
        """
        底层测试驱动器
        :param xml_path: XML 文件路径
        :param model_path: RL 模型路径 (如果 mode='hybrid' 需提供)
        :param mode: 控制模式 'pcc' 或 'hybrid'
        :param render: 是否开启实时可视化
        """
        self.xml_path = xml_path
        self.mode = mode
        self.render = render
        self.model_loaded = False

        # --- 1. 物理参数 (需与训练/Control保持一致) ---
        self.params = {
            "bend_stiffness": 5e8,
            "twist_stiffness": 1e11,
            "joint_damping": 30.0,
            "actuator_kp": 800.0,
        }

        # --- 2. 加载 MuJoCo (带参数格式化) ---
        self._load_mujoco()

        # --- 3. 核心对象初始化 ---
        # 目标点 (默认悬停在上方)
        self.current_target = np.array([0.0, 0.0, 1.1])
        # 外部负载 (默认无)
        self.current_force = np.zeros(3)

        # 机器人数学模型 & 控制器
        self.robot_math_model = SoftRobotModel(
            L_list=[0.5, 0.5], r_disk=0.08, base_pos=np.array([0, 0, 0.3])
        )
        self.controller = PCCController(
            self.robot_math_model, dt=self.mj_model.opt.timestep
        )

        # RL 模型
        self.residual_scale = 0.08
        self.mid_body_name = "ring_5_body"
        self.rl_model = None
        self.last_tip_pos = np.zeros(3)
        self.current_tip_velocity = np.zeros(3)

        if self.mode == "hybrid":
            if model_path:
                print(f"[Tester] Loading RL Model from {model_path}...")
                try:
                    # [修改点] 强制指定 device="cpu"
                    self.rl_model = PPO.load(model_path, device="cpu")
                    self.model_loaded = True
                except Exception as e:
                    print(f"[Warning] Failed to load RL model: {e}. Fallback to PCC.")
                    self.mode = "pcc"
            else:
                print(
                    "[Warning] No model_path provided for hybrid mode. Fallback to PCC."
                )
                self.mode = "pcc"

        # --- 4. 冻结 PCC 逻辑 (新增) ---
        self.freeze_pcc = False
        self.frozen_l_base = None
        self.frozen_expected_mid_pos = None

        # --- 4. 可视化 ---
        self.video = video
        self.obs_type = obs_type
        if video:
            self.plotter = RealTimePlotter()
        self.viewer = None
        if self.render:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

    def stabilize_at(self, target_pos, tolerance=0.00, max_steps=800):
        """
        [新增功能] 让机器人在后台运行直到稳定在目标点
        :param target_pos: 目标位置
        :param tolerance: 允许的误差阈值 (米)
        :param max_steps: 最大预热步数，防止死循环
        """
        print(f"[Tester] Stabilizing at {target_pos}...")
        self.set_target(target_pos)

        # 暂时关闭渲染以加速
        old_render_state = self.render
        # 注意：这里我们不关闭 viewer，只是不调用 sync，这样速度最快
        self.warm = 1
        for i in range(max_steps):
            # 1. 正常步进，但不加额外扰动（除了重力）
            # 确保负载归零，让它自然稳定
            self.set_load(np.zeros(3))

            # 2. 调用 step 获取误差
            info = self.step()

            # 3. 检查收敛
            if info["error"] < tolerance:
                # 额外多跑50步以消除由于动量带来的微小震荡
                for _ in range(50):
                    self.step()
                print(f"[Tester] Stabilized at step {i} (Error: {info['error']:.4f}m)")
                break
        else:
            print(
                f"[Tester] Warning: Failed to converge within {max_steps} steps. Final Error: {info['error']:.4f}m"
            )
        self.warm = 0
        # === 关键：重置仿真时间为 0 ===
        # 这样你的测试脚本就会认为实验是从 0 秒开始的，且机器人已经就位
        import contextlib
        ctx = self.viewer.lock() if self.viewer else contextlib.nullcontext()
        with ctx:
            self.mj_data.time = 0.0

    def _load_mujoco(self):
        """内部函数：读取模板并初始化物理引擎"""
        with open(self.xml_path, "r") as f:
            xml_template = f.read()

        xml_content = xml_template.format(
            bend_val=f"{self.params['bend_stiffness']:.1e}",
            twist_val=f"{self.params['twist_stiffness']:.1e}",
            damping_val=f"{self.params['joint_damping']:.1f}",
            kp_val=f"{self.params['actuator_kp']:.1f}",
        )

        self.mj_model = mujoco.MjModel.from_xml_string(xml_content)
        self.mj_data = mujoco.MjData(self.mj_model)

        # 缓存 ID
        self.tip_body_name = "ring_10_body"
        self.tip_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.tip_body_name
        )

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
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in act_names
        ]

        # Retrieve body IDs for all 10 rings (assumes naming convention "ring_1_body" to "ring_10_body")
        self.ring_body_ids = []
        for i in range(1, 11):
            body_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_BODY, f"ring_{i}_body"
            )
            if body_id != -1:
                self.ring_body_ids.append(body_id)

    def _get_obs(self, target_pos, l_base):
        """内部函数：构建 RL 观测向量"""
        try:
            real_tip_pos = self.mj_data.site("rod_tip").xpos.copy()
        except:
            real_tip_pos = self.mj_data.body(self.tip_body_name).xpos.copy()

        # 计算特征
        tip_error = target_pos - real_tip_pos

        if self.obs_type == "tip_velocity":
            if not hasattr(self, 'current_tip_velocity'):
                self.current_tip_velocity = np.zeros(3)
            feature_3 = self.current_tip_velocity.copy()
        else:
            try:
                feature_3 = self.mj_data.body(self.mid_body_name).xpos.copy()
            except:
                feature_3 = np.zeros(3)

        # 14维向量: [TipErr(3), Feature(3), PCC_Cmd(8)]
        obs = np.concatenate([tip_error, feature_3, l_base]).astype(np.float32)
        return obs

    # =========================================================
    #  用户接口 API
    # =========================================================

    def freeze_pcc_control(self, freeze=True):
        """
        冻结/解冻 PCC 的控制输出。
        当 freeze=True 时，PCC 的输出 l_base 将固定为当前时刻的值，不再随误差变化。
        """
        self.freeze_pcc = freeze
        if not freeze:
            self.frozen_l_base = None
            self.frozen_expected_mid_pos = None
            print("[Tester] PCC control un-frozen.")
        else:
            print("[Tester] PCC control frozen.")

    def reset(self):
        """重置仿真环境和控制器状态"""
        if self.viewer:
            with self.viewer.lock():
                mujoco.mj_resetData(self.mj_model, self.mj_data)
                self.mj_data.xfrc_applied.fill(0)
        else:
            mujoco.mj_resetData(self.mj_model, self.mj_data)
            self.mj_data.xfrc_applied.fill(0)
            
        self.last_tip_pos = np.zeros(3)
        self.current_tip_velocity = np.zeros(3)
        if self.video:
            self.plotter = RealTimePlotter()
        self.controller = PCCController(
            self.robot_math_model, dt=self.mj_model.opt.timestep
        )
        self.freeze_pcc = False
        self.frozen_l_base = None
        self.frozen_expected_mid_pos = None
        if self.viewer:
            self.viewer.sync()

    def set_target(self, target_pos):
        """
        修改当前目标点
        :param target_pos: [x, y, z] 列表或数组
        """
        self.current_target = np.array(target_pos, dtype=np.float64)
        # 更新可视化小球
        if self.viewer:
            with self.viewer.lock():
                self.mj_data.mocap_pos[0] = self.current_target
        else:
            self.mj_data.mocap_pos[0] = self.current_target

    def set_load(self, force_vec):
        """
        修改施加在末端的负载/扰动
        :param force_vec: [Fx, Fy, Fz] 单位牛顿
        """
        self.current_force = np.array(force_vec, dtype=np.float64)

    def step(self):
        """
        执行单步仿真
        :return: info 字典，包含时间、实际位置、误差、控制指令等详细信息
        """
        import contextlib
        ctx = self.viewer.lock() if self.viewer else contextlib.nullcontext()
        
        with ctx:
            # 1. 施加外力 (Force Application)
            self.mj_data.xfrc_applied.fill(0)
            if self.tip_body_id != -1:
                self.mj_data.xfrc_applied[self.tip_body_id][:3] = self.current_force
    
            # 2. 获取反馈 (Feedback)
            try:
                curr_tip = self.mj_data.site("rod_tip").xpos.copy()
            except:
                curr_tip = self.mj_data.body(self.tip_body_name).xpos.copy()
                
            if not hasattr(self, 'last_tip_pos') or np.linalg.norm(self.last_tip_pos) == 0:
                self.last_tip_pos = curr_tip.copy()
            self.current_tip_velocity = (curr_tip - self.last_tip_pos) / self.mj_model.opt.timestep
            self.last_tip_pos = curr_tip.copy()
    
            # 3. 计算基准控制量 (PCC Control)
            l_base, _ = self.controller.step(self.current_target, curr_tip)
    
            if self.freeze_pcc:
                if self.frozen_l_base is None:
                    self.frozen_l_base = l_base.copy()
                else:
                    l_base = self.frozen_l_base.copy()
            else:
                self.frozen_l_base = l_base.copy()
    
            # 4. 计算残差 (RL Logic)
            l_residual = np.zeros(8)
            if self.mode == "hybrid" and self.model_loaded:
                obs = self._get_obs(self.current_target, l_base)
                action, _ = self.rl_model.predict(obs, deterministic=True)
                l_residual = action * self.residual_scale
    
            # 5. 合成并下发指令 (Command limit & Actuation)
            l_cmd = l_base + l_residual
            l_cmd[:4] = np.clip(l_cmd[:4], 0.1, 0.9)
            l_cmd[4:] = np.clip(l_cmd[4:], 0.3, 1.7)
    
            self.mj_data.ctrl[self.act_ids] = l_cmd
            error = np.linalg.norm(self.current_target - curr_tip)
    
            # 3. 每隔 N 步更新一次图形（避免降低仿真频率）
            if self.video and self.warm == 0:
                self.plotter.update(error)
            # 6. 物理步进 (Physics Step)
            mujoco.mj_step(self.mj_model, self.mj_data)
                
        # 7. 渲染 (Render)
        if self.viewer:
            self.viewer.sync()

        # 计算最大段间弯曲角度来检测异常折叠 (Buckling)
        max_angle_deg = 0.0
        if len(self.ring_body_ids) > 1:
            for i in range(len(self.ring_body_ids) - 1):
                id_curr = self.ring_body_ids[i]
                id_next = self.ring_body_ids[i + 1]
                
                # 获取四元数
                quat_curr = self.mj_data.xquat[id_curr]
                quat_next = self.mj_data.xquat[id_next]
                
                # 获取旋转矩阵并提取局部 Z 轴在全局坐标系中的方向
                mat_curr = np.zeros(9)
                mat_next = np.zeros(9)
                mujoco.mju_quat2Mat(mat_curr, quat_curr)
                mujoco.mju_quat2Mat(mat_next, quat_next)
                
                z_curr = mat_curr.reshape(3, 3)[:, 2]
                z_next = mat_next.reshape(3, 3)[:, 2]
                
                # 计算两个 Z 轴向量的夹角
                dot_product = np.clip(np.dot(z_curr, z_next), -1.0, 1.0)
                angle_rad = np.arccos(dot_product)
                angle_deg = np.degrees(angle_rad)
                
                if angle_deg > max_angle_deg:
                    max_angle_deg = angle_deg

        # 8. 返回数据
        return {
            "time": self.mj_data.time,
            "target": self.current_target.copy(),
            "current_pos": curr_tip,
            "error": np.linalg.norm(self.current_target - curr_tip),
            "force": self.current_force.copy(),
            "cmd_total": l_cmd,
            "cmd_pcc": l_base,
            "cmd_rl": l_residual,
            "max_bending_angle_deg": max_angle_deg,
            "xi_curr": self.controller.xi_curr.copy(),
        }

    def close(self):
        if self.viewer:
            self.viewer.close()
