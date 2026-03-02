import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import mujoco
import mujoco.viewer
import numpy as np
from numba import njit
import time
import matplotlib.pyplot as plt


# ==========================================
# 1. Numba 加速的正运动学 (FK) 与 雅可比 (Jacobian)
# ==========================================
@njit(fastmath=True, cache=True)
def _calc_fk_jit(xi, L_arr, base_pos):
    """
    计算两段 PCC 的末端位置 (适配：倒置且无基座旋转的情况)
    逻辑：为了保持右手系且向下生长，我们定义局部坐标系为：
    Local X = World X (不变)
    Local Y = -World Y (反向)
    Local Z = -World Z (反向，向下)
    """
    # 初始化变换矩阵
    T = np.zeros((4, 4))

    # --- 关键修正 ---
    # 绕 X 轴旋转 180 度
    T[0, 0] = 1.0  # X 轴方向不变！(这样拉东边就是去东边)
    T[1, 1] = -1.0  # Y 轴反向
    T[2, 2] = -1.0  # Z 轴反向 (向下生长)
    T[3, 3] = 1.0

    # 设置基座位置 (比如 1.3m)
    T[:3, 3] = base_pos

    # 循环计算每一段 (Section)
    for i in range(2):
        kx = xi[2 * i]
        ky = xi[2 * i + 1]
        L = L_arr[i]

        theta = np.sqrt(kx**2 + ky**2)
        T_sec = np.eye(4)

        if theta < 1e-6:
            # 奇异点处理：直线
            T_sec[2, 3] = L
        else:
            phi = np.arctan2(ky, kx)
            r = L / theta  # 曲率半径

            c_phi = np.cos(phi)
            s_phi = np.sin(phi)
            c_th = np.cos(theta)
            s_th = np.sin(theta)

            # 论文中标准的 PCC 运动学
            dx = r * (1 - c_th) * c_phi
            dy = r * (1 - c_th) * s_phi
            dz = r * s_th

            # 旋转矩阵构建
            Rz = np.array([[c_phi, -s_phi, 0.0], [s_phi, c_phi, 0.0], [0.0, 0.0, 1.0]])
            Ry = np.array([[c_th, 0.0, s_th], [0.0, 1.0, 0.0], [-s_th, 0.0, c_th]])
            R_sec = Rz @ Ry @ Rz.T

            T_sec[:3, :3] = R_sec
            T_sec[0, 3] = dx
            T_sec[1, 3] = dy
            T_sec[2, 3] = dz

        T = T @ T_sec

    return T


@njit(fastmath=True, cache=True)
def _calc_jacobian_jit(xi, L_arr, base_pos, eps=1e-4):
    """数值差分计算雅可比 J (3x4)"""
    J = np.zeros((3, 4))
    T0 = _calc_fk_jit(xi, L_arr, base_pos)
    p0 = T0[:3, 3]

    for i in range(4):
        xi_p = xi.copy()
        xi_p[i] += eps
        Tp = _calc_fk_jit(xi_p, L_arr, base_pos)
        pp = Tp[:3, 3]
        J[:, i] = (pp - p0) / eps
    return J


@njit(fastmath=True, cache=True)
def _calc_fk_points_jit(xi, L_arr, base_pos):
    """
    修改后的FK：返回 [mid_pos, tip_pos]
    用于可视化
    """
    T = np.zeros((4, 4))
    # 保持与 FK 一致的初始化
    T[0, 0] = 1.0
    T[1, 1] = -1.0
    T[2, 2] = -1.0
    T[3, 3] = 1.0
    T[:3, 3] = base_pos

    points = np.zeros((2, 3))

    for i in range(2):
        kx = xi[2 * i]
        ky = xi[2 * i + 1]
        L = L_arr[i]

        theta = np.sqrt(kx**2 + ky**2)
        T_sec = np.eye(4)
        if theta < 1e-6:
            T_sec[2, 3] = L
        else:
            phi = np.arctan2(ky, kx)
            r = L / theta
            c_phi = np.cos(phi)
            s_phi = np.sin(phi)
            c_th = np.cos(theta)
            s_th = np.sin(theta)

            dx = r * (1 - c_th) * c_phi
            dy = r * (1 - c_th) * s_phi
            dz = r * s_th

            Rz = np.array([[c_phi, -s_phi, 0.0], [s_phi, c_phi, 0.0], [0.0, 0.0, 1.0]])
            Ry = np.array([[c_th, 0.0, s_th], [0.0, 1.0, 0.0], [-s_th, 0.0, c_th]])
            R_sec = Rz @ Ry @ Rz.T

            T_sec[:3, :3] = R_sec
            T_sec[0, 3] = dx
            T_sec[1, 3] = dy
            T_sec[2, 3] = dz

        T = T @ T_sec
        points[i, :] = T[:3, 3]

    return points


# ==========================================
# 2. 修正后的机器人数学模型
# ==========================================
class SoftRobotModel:
    def __init__(self, L_list, r_disk, base_pos):
        self.L = np.array(L_list, dtype=np.float64)
        self.r = float(r_disk)
        self.base_pos = np.array(base_pos, dtype=np.float64)

        # 预计算驱动映射矩阵
        self._init_actuation_matrices()

    def get_fk_points(self, xi):
        return _calc_fk_points_jit(xi, self.L, self.base_pos)

    def _init_actuation_matrices(self):
        """
        构建配置空间 xi 到 驱动空间 l 的线性映射
        注意：因为我们定义 Local Y = -World Y，所以必须调整线缆的角度
        """
        # --- 1. 重新定义线缆角度 (适配 diag(1, -1, -1)) ---

        # Group 1: North, South, East, West (XML order)
        # North (World +Y) -> Local -Y -> Angle -pi/2
        # South (World -Y) -> Local +Y -> Angle +pi/2
        # East  (World +X) -> Local +X -> Angle 0
        # West  (World -X) -> Local -X -> Angle pi
        angles_g1 = np.array([-np.pi / 2, np.pi / 2, 0, np.pi])

        # Group 2: First, Second, Third, Fourth
        # First (World +X, +Y) -> Local (+X, -Y) -> -pi/4
        # Second (World -X, +Y) -> Local (-X, -Y) -> -3pi/4
        # Third (World -X, -Y) -> Local (-X, +Y) -> +3pi/4
        # Fourth (World +X, -Y) -> Local (+X, +Y) -> +pi/4
        angles_g2 = np.array([-np.pi / 4, -3 * np.pi / 4, 3 * np.pi / 4, np.pi / 4])

        # 计算每根线缆在局部坐标系下的力臂 (delta向量)
        delta_1 = self.r * np.vstack([np.cos(angles_g1), np.sin(angles_g1)])
        delta_2 = self.r * np.vstack([np.cos(angles_g2), np.sin(angles_g2)])

        Psi = np.zeros((8, 4))

        # --- 填充 Group 1 ---
        for i in range(4):
            x, y = delta_1[:, i]
            Psi[i, 0] = -x
            Psi[i, 1] = -y

        # --- 填充 Group 2 ---
        for i in range(4):
            idx = i + 4
            x, y = delta_2[:, i]
            Psi[idx, 0] = -x
            Psi[idx, 1] = -y
            Psi[idx, 2] = -x
            Psi[idx, 3] = -y

        self.Psi = Psi
        self.Psi_pinv = np.linalg.pinv(Psi)

    def get_fk(self, xi):
        return _calc_fk_jit(xi, self.L, self.base_pos)

    def get_jacobian(self, xi):
        return _calc_jacobian_jit(xi, self.L, self.base_pos)

    def config_to_actuator(self, xi):
        return self.Psi @ xi


# ==========================================
# 3. 控制器 (逻辑不变，参数微调)
# ==========================================
class PCCController:
    def __init__(self, model: SoftRobotModel, dt=0.002):
        self.model = model
        self.dt = dt
        self.xi_curr = np.zeros(4)

        # 增加初始张力补偿 (重力会让机器人伸长，需要适当收紧)
        # XML 中线缆初始是直的，但重力会拉长
        self.l_rest = np.array([0.5] * 4 + [1.0] * 4)

        self.K_inv = 2.0
        self.damping = 1e-2

    def step(self, target_pos, current_tip_pos):
        # 1. 计算误差
        error = target_pos - current_tip_pos

        # 2. 获取雅可比
        J = self.model.get_jacobian(self.xi_curr)

        # 3. 求解 d_xi
        H = J.T @ J + self.damping * np.eye(4)
        g = J.T @ (self.K_inv * error)
        d_xi = np.linalg.solve(H, g)

        d_xi = np.clip(d_xi, -15.0, 15.0)  # 放宽一点速度限制

        # 4. 积分
        self.xi_curr += d_xi * 5.0 * self.dt
        self.xi_curr = np.clip(self.xi_curr, -18.0, 18.0)  # 稍微放宽曲率限制

        # 5. 输出
        delta_l = self.model.config_to_actuator(self.xi_curr)
        l_cmd = self.l_rest + delta_l

        return l_cmd, np.linalg.norm(error)


# ==========================================
# 4. 轨迹生成器 (适配基座高度)
# ==========================================
class TrajectoryGenerator:
    def __init__(self, traj_type="rose", center_z=0.3, radius=0.15, speed=0.1):
        """
        center_z: 轨迹中心高度。
        基座在 1.3m，臂长 1.0m，所以自然下垂点大约在 0.3m 左右。
        """
        self.type = traj_type
        self.center_z = center_z
        self.radius = radius
        self.speed = speed
        self.phase_offset = 0.0

    def reset(self):
        self.phase_offset = np.random.uniform(0, 2 * np.pi)

    def get_target(self, t):
        phase = self.speed * t + self.phase_offset
        if self.type == "circle":
            x = self.radius * np.cos(phase)
            y = self.radius * np.sin(phase)
            z = self.center_z
        elif self.type == "rose":
            k = 3
            r = self.radius * np.cos(k * phase)
            x = r * np.cos(phase)
            y = r * np.sin(phase)
            z = self.center_z
        else:
            x, y, z = 0, 0, self.center_z

        return np.array([x, y, z])


# ==========================================
# 5. 仿真主循环
# ==========================================
def run_simulation():
    XML_PATH = "./assets/two_disks_uj.xml"

    # 读取 XML
    with open(XML_PATH, "r") as f:
        xml_template = f.read()
        xml_content = xml_template.format(
            bend_val="5e8", twist_val="1e11", damping_val="30.0", kp_val="800.0"
        )

    model = mujoco.MjModel.from_xml_string(xml_content)
    data = mujoco.MjData(model)

    # === 关键配置 ===
    # 假设 XML 中 <body name="rod_base" pos="0 0 1.3">
    # 且没有 rotation
    BASE_HEIGHT = 1.3

    # 1. 实例化数学模型
    # 注意：这里会使用新的 _init_actuation_matrices 自动处理坐标反转
    robot_model = SoftRobotModel(
        L_list=[0.5, 0.5], r_disk=0.08, base_pos=[0, 0, BASE_HEIGHT]
    )

    # 2. 控制器
    controller = PCCController(robot_model, dt=model.opt.timestep)

    # 3. 轨迹生成：在 0.3m 高度画圆
    traj_gen = TrajectoryGenerator(
        traj_type="circle", center_z=0.3, radius=0.2, speed=0.5
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
    act_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in act_names
    ]

    history = {"target": [], "actual": [], "time": [], "error": []}

    print("[Info] Start Simulation (Direct Inverted Mode)...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        while viewer.is_running():
            step_start = time.time()
            t = step_start - start_time

            # 1. 目标
            target = [0.4, 0, 0.5]
            data.mocap_pos[0] = target

            # 2. 反馈
            try:
                tip_pos = data.site("rod_tip").xpos.copy()
            except:
                tip_pos = data.body("ring_10_body").xpos.copy()

            # 3. 控制
            l_cmd, err = controller.step(target, tip_pos)

            # 4. 指令下发 (安全限幅)
            # l_cmd[:4] = np.clip(l_cmd[:4], 0.1, 0.9)
            # l_cmd[4:] = np.clip(l_cmd[4:], 0.4, 1.6)
            data.ctrl[act_ids] = l_cmd

            # 5. 物理步进
            mujoco.mj_step(model, data)
            viewer.sync()

            history["target"].append(target)
            history["actual"].append(tip_pos)
            history["time"].append(t)
            history["error"].append(err)

            time_until_next = model.opt.timestep - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)

    plot_results(history)


def plot_results(h):
    target = np.array(h["target"])
    actual = np.array(h["actual"])
    t = np.array(h["time"])
    err = np.array(h["error"])

    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(target[:, 0], target[:, 1], "g--", label="Ref")
    plt.plot(actual[:, 0], actual[:, 1], "r", label="Act")
    plt.title("XY Path (Should match perfectly now)")
    plt.axis("equal")
    plt.legend()

    plt.subplot(122)
    plt.plot(t, err)
    plt.title("Error")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run_simulation()
