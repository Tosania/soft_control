import numpy as np
import mujoco
from src.core.tester import SoftRobotTester

class FuzzyScaleController:
    """
    轻量级模糊比例控制器 (Fuzzy Logic Scale Controller)
    """
    def __init__(self, base_scale=0.08, min_scale=0.0, max_scale=0.14):
        self.base_scale = base_scale
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.prev_error = 0.0
        self.smoothed_ec = 0.0 # 平滑后的误差变化率
        self.smoothed_scale = base_scale # 平滑后的最终缩放因子
        
        # 增大低通滤波系数以减小延迟 (延迟是导致超调的主要原因)
        self.alpha_ec = 0.4    # 误差变化率滤波 (之前是 0.1)
        self.alpha_scale = 0.3 # 输出滤波 (之前是 0.05)
        
    def get_scale(self, current_error, dt=0.01):
        # 计算误差变化率 (Error Change)
        raw_ec = (current_error - self.prev_error) / dt
        self.prev_error = current_error
        
        # 对误差变化率进行低通滤波
        self.smoothed_ec = self.alpha_ec * raw_ec + (1 - self.alpha_ec) * self.smoothed_ec
        error_change = self.smoothed_ec
        
        # 将误差变化率 EC 归一化到 [-1, 1] 区间，0.4m/s 视为非常大的发散/收敛速度
        ec_factor = np.clip(error_change / 0.4, -1.0, 1.0)
        
        # —— 极简防超调规则 ——
        if ec_factor > 0:
            # 误差正在变大 (如受到外部扰动，或者刚刚发生超调时的发散)
            # 误差变大越快，给的 RL 补偿越大，阻止其偏离
            drive = ec_factor 
            scale = self.base_scale + (self.max_scale - self.base_scale) * drive
        else:
            # 误差正在变小 (正朝着目标移动)
            # 如果收敛速度很快 (abs(ec_factor) 很大)，主动减小 RL 的干预以防「推得太猛」导致冲过头
            # 如果收敛速度缓慢，恢复到 base_scale 进行微调
            reduction_drive = abs(ec_factor)
            scale = self.base_scale - (self.base_scale - self.min_scale) * reduction_drive
            
        raw_scale = np.clip(scale, self.min_scale, self.max_scale)
        
        # 输出滤波，系数增大使得响应更快，不会将很大的 scale 滞留到穿过目标点的时候
        self.smoothed_scale = self.alpha_scale * raw_scale + (1 - self.alpha_scale) * self.smoothed_scale
            
        return float(self.smoothed_scale)

class AdaptiveSoftRobotTester(SoftRobotTester):
    """
    带有模糊自适应 residual_scale 调节的底层测试驱动器
    """
    def __init__(self, xml_path, model_path=None, mode="hybrid", render=True, video=False):
        super().__init__(xml_path, model_path, mode, render, video)
        
        # 初始化模糊控制器
        self.fuzzy_controller = FuzzyScaleController(
            base_scale=0.08, 
            min_scale=0.0, 
            max_scale=0.14
        )
        self.current_residual_scale = self.fuzzy_controller.base_scale
        # 补全父类中可能缺少的初始化属性
        if not hasattr(self, 'warm'):
            self.warm = 0

    def reset(self):
        super().reset()
        self.fuzzy_controller.prev_error = 0.0
        self.current_residual_scale = self.fuzzy_controller.base_scale

    def step(self):
        """
        重写 step 方法以引入自适应 residual_scale
        """
        # 1. 施加外力 (Force Application)
        self.mj_data.xfrc_applied.fill(0)
        if self.tip_body_id != -1:
            self.mj_data.xfrc_applied[self.tip_body_id][:3] = self.current_force

        # 2. 获取反馈 (Feedback)
        try:
            curr_tip = self.mj_data.site("rod_tip").xpos.copy()
        except:
            curr_tip = self.mj_data.body(self.tip_body_name).xpos.copy()

        error = np.linalg.norm(self.current_target - curr_tip)

        # 3. 计算基准控制量 (PCC Control)
        l_base, _ = self.controller.step(self.current_target, curr_tip)

        if self.freeze_pcc:
            if self.frozen_l_base is None:
                self.frozen_l_base = l_base.copy()
            else:
                l_base = self.frozen_l_base.copy()
        else:
            self.frozen_l_base = l_base.copy()

        # [新增核心逻辑]：根据误差自适应调节 residual_scale
        dt = self.mj_model.opt.timestep
        self.current_residual_scale = self.fuzzy_controller.get_scale(error, dt)

        # 4. 计算残差 (RL Logic)
        l_residual = np.zeros(8)
        if self.mode == "hybrid" and self.model_loaded:
            obs = self._get_obs(self.current_target, l_base)
            action, _ = self.rl_model.predict(obs, deterministic=True)
            # 使用自适应计算出的 scale
            l_residual = action * self.current_residual_scale

        # 5. 合成并下发指令 (Command limit & Actuation)
        l_cmd = l_base + l_residual
        l_cmd[:4] = np.clip(l_cmd[:4], 0.1, 0.9)
        l_cmd[4:] = np.clip(l_cmd[4:], 0.3, 1.7)

        self.mj_data.ctrl[self.act_ids] = l_cmd
        
        # 视频绘图更新
        if self.video and getattr(self, 'warm', 0) == 0:
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
                
                quat_curr = self.mj_data.xquat[id_curr]
                quat_next = self.mj_data.xquat[id_next]
                
                mat_curr = np.zeros(9)
                mat_next = np.zeros(9)
                mujoco.mju_quat2Mat(mat_curr, quat_curr)
                mujoco.mju_quat2Mat(mat_next, quat_next)
                
                z_curr = mat_curr.reshape(3, 3)[:, 2]
                z_next = mat_next.reshape(3, 3)[:, 2]
                
                dot_product = np.clip(np.dot(z_curr, z_next), -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(dot_product))
                
                if angle_deg > max_angle_deg:
                    max_angle_deg = angle_deg

        # 8. 返回数据
        return {
            "time": self.mj_data.time,
            "target": self.current_target.copy(),
            "current_pos": curr_tip,
            "error": error,
            "force": self.current_force.copy(),
            "cmd_total": l_cmd,
            "cmd_pcc": l_base,
            "cmd_rl": l_residual,
            "residual_scale": self.current_residual_scale, # 输出自适应 Scale 供日志记录
            "max_bending_angle_deg": max_angle_deg,
            "xi_curr": self.controller.xi_curr.copy(),
        }
