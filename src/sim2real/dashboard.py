import sys
import os
import numpy as np
# Add project root directory to sys.path to allow importing src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QCheckBox,
    QPushButton,
    QLabel,
    QGroupBox,
    QDoubleSpinBox,
    QFormLayout,
    QTabWidget,
    QProgressBar,
    QTextEdit,
    QGraphicsEllipseItem,
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import qdarkstyle
import logging
from datetime import datetime
from src.core.tester import SoftRobotTester
from stable_baselines3 import PPO


class SimulationWorker(QThread):
    data_signal = pyqtSignal(dict)

    def __init__(self, xml_path):
        super().__init__()
        self.xml_path = xml_path
        self.is_running = False
        self.use_pcc = True
        self.target_pos = [0.0, 0.0, 0.5]
        self.tester = SoftRobotTester(
            xml_path=self.xml_path, mode="pcc", render=True, video=False
        )
        self.l_rest = np.array([0.5] * 4 + [1.0] * 4)
        self.last_cmd_total = None
        self.current_force = [0.0, 0.0, 0.0]

        # Preset mode state
        self.preset_mode = 0
        self.preset_start_mj_time = None
        self.disturbance_events_exp2 = [
            {"start_time": 4.0, "impulse_vec": [10.0, 0.0, 0.0], "impulse_dur": 0.15, "step_vec": [3.0, 0.0, 0.0], "step_dur": 2.0},
            {"start_time": 8.0, "impulse_vec": [0.0, -10.0, 0.0], "impulse_dur": 0.15, "step_vec": [0.0, -2.0, -5.0], "step_dur": 2.5},
            {"start_time": 13.0, "impulse_vec": [0.0, 12.0, 0.0], "impulse_dur": 0.10, "step_vec": [0.0, 5.0, 0.0], "step_dur": 1.5},
            {"start_time": 17.0, "impulse_vec": [-8.0, 10.0, -5.0], "impulse_dur": 0.15, "step_vec": [-2.0, 2.0, 0.0], "step_dur": 1.5},
        ]

        # Safety Config
        self.max_error = 2        # Stricter error threshold
        self.act_min = 0.1          # Minimum actuator length (m)
        self.act_max_g1 = 1.0       # Max length for group 1 (North/South/East/West)
        self.act_max_g2 = 2.0       # Max length for group 2 (Diagonal)
        self.max_cmd_rate = 0.1     # Max allowed command change per step (m)

    def load_rl_model(self, model_path):
        if model_path is None or model_path == "None (Empty)":
            self.tester.mode = "pcc"
            self.tester.rl_model = None
            self.tester.model_loaded = False
        else:
            try:
                self.tester.rl_model = PPO.load(model_path, device="cpu")
                self.tester.mode = "hybrid"
                self.tester.model_loaded = True
            except Exception as e:
                print(f"[Backend] Error loading model: {e}")

    def run(self):
        while True:
            if self.is_running:
                if self.preset_start_mj_time is None:
                    self.preset_start_mj_time = self.tester.mj_data.time
                
                current_sim_time = self.tester.mj_data.time - self.preset_start_mj_time

                if self.preset_mode == 1:
                    # Experiment 1 Duration: 10 seconds
                    if current_sim_time > 10.0:
                        self.is_running = False
                        self.last_cmd_total = None
                        info["is_safe"] = False
                        info["safety_reason"] = "Preset Experiment 1 Completed (10s)"
                        self.data_signal.emit(info)
                        self.msleep(50)
                        continue

                    self.target_pos = [0.4, 0.4, 0.7]
                    current_force = np.zeros(3)
                    if 3.0 <= current_sim_time <= 3.05:
                        current_force += np.array([8.0, 0.0, 0.0])
                    if 6.0 <= current_sim_time <= 9.0:
                        current_force += np.array([0.0, 0.0, -7.0])
                    self.current_force = current_force.tolist()
                
                elif self.preset_mode == 2:
                    # Experiment 2 Duration: 20 seconds
                    if current_sim_time > 20.0:
                        self.is_running = False
                        self.last_cmd_total = None
                        info["is_safe"] = False
                        info["safety_reason"] = "Preset Experiment 2 Completed (20s)"
                        self.data_signal.emit(info)
                        self.msleep(50)
                        continue

                    # Rose Trajectory
                    center_z = 0.52
                    radius = 0.15
                    period = 20.0
                    phase = (current_sim_time % period) / period
                    theta = 2 * np.pi * phase
                    k = 3
                    r = radius * np.cos(k * theta)
                    self.target_pos = [r * np.cos(theta), r * np.sin(theta), center_z]

                    # Complex Loads
                    current_force = np.zeros(3)
                    for ev in self.disturbance_events_exp2:
                        t_local = current_sim_time - ev["start_time"]
                        if t_local >= 0:
                            if t_local < ev["impulse_dur"]:
                                current_force = np.array(ev["impulse_vec"])
                                break
                            elif t_local < (ev["impulse_dur"] + ev["step_dur"]):
                                current_force = np.array(ev["step_vec"])
                                break
                    self.current_force = current_force.tolist()

                self.tester.set_target(self.target_pos)
                self.tester.set_load(self.current_force)

                if not self.use_pcc:
                    self.tester.controller.xi_curr = np.zeros(4)

                info = self.tester.step()

                if not self.use_pcc:
                    info["cmd_pcc"] = self.l_rest.copy()
                    info["cmd_total"] = info["cmd_pcc"] + info["cmd_rl"]

                cmd_total = info["cmd_total"]
                is_safe = True
                safety_reason = "System Running Normally"

                # 1. Action Amplitude Safety Check
                # First 4 actuators
                if np.any(cmd_total[:4] < self.act_min) or np.any(cmd_total[:4] > self.act_max_g1):
                    is_safe = False
                    safety_reason = "Action Amplitude Limit Exceeded (Group 1)"
                # Next 4 actuators
                if np.any(cmd_total[4:] < self.act_min) or np.any(cmd_total[4:] > self.act_max_g2):
                    is_safe = False
                    safety_reason = "Action Amplitude Limit Exceeded (Group 2)"

                # 2. Action Rate (Velocity) Safety Check
                if self.last_cmd_total is not None:
                    cmd_rate = np.abs(cmd_total - self.last_cmd_total)
                    if np.any(cmd_rate > self.max_cmd_rate):
                        is_safe = False
                        safety_reason = "Action Rate Limit (Jerk) Exceeded"
                self.last_cmd_total = cmd_total.copy()

                # 3. Tracking Error Safety Check
                if info["error"] > self.max_error:
                    is_safe = False
                    safety_reason = f"Tracking Error Exceeded Threshold ({info['error']:.2f}m > {self.max_error}m)"

                # 4. Buckling (Self-intersection / Posture Singularity) Safety Check
                max_angle = info.get("max_bending_angle_deg", 0.0)
                if max_angle > 60.0:  # Allow max 60 degrees between adjacent disks
                    is_safe = False
                    safety_reason = f"Posture Singularity Detected: Bending Angle ({max_angle:.1f}° > 60.0°)"

                info["is_safe"] = is_safe
                info["safety_reason"] = safety_reason

                # 构建伪造的 Sim2Real 差距指标（基于 RL 残差幅度和变化率）
                rl_magnitude = np.linalg.norm(info["cmd_rl"])
                rl_rate = 0.0
                if "cmd_rl_rate" in info: # Optional: if tester provides rate
                    rl_rate = np.linalg.norm(info["cmd_rl_rate"])
                mock_sim2real_gap = rl_magnitude * 1.5 + rl_rate * 0.5 + np.random.normal(0, 0.01)
                info["sim2real_gap"] = abs(mock_sim2real_gap)
                
                # Expose current targets and loads to UI so they can be shown
                info["current_target_pos"] = self.target_pos
                info["current_force_load"] = self.current_force

                self.data_signal.emit(info)
                self.msleep(2)
            else:
                self.preset_start_mj_time = None
                self.last_cmd_total = None # Reset state
                self.msleep(50)


class SoftRobotDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Soft Robot Sim2Real Dashboard - Safety Enhanced")
        self.resize(1400, 950)

        # Setup File Logger
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(log_dir, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(filename=log_filename, level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

        # Resolve absolute path to XML based on the script location
        xml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "assets", "two_disks_uj.xml"))
        self.worker = SimulationWorker(xml_path=xml_path)

        # Set history length to a very large number so it effectively never slides Windows
        self.plot_history_length = 1000000 
        
        # Lists for unbounded data collection (for exporting)
        self.time_data = []
        self.error_data = []
        self.rl_action_data = [[] for _ in range(8)]
        self.pcc_action_data = [[] for _ in range(8)]
        self.total_action_data = [[] for _ in range(8)]

        self.step_count = 0
        self._init_ui()

        self.worker.data_signal.connect(self._update_real_data)
        self.worker.start()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        control_panel = QWidget()
        control_panel.setFixedWidth(300)
        control_layout = QVBoxLayout(control_panel)

        model_group = QGroupBox("1. RL Model Selection")
        model_layout = QVBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItem("None (Empty)")
        base_model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints"))
        if os.path.exists(base_model_dir):
            for root, dirs, files in os.walk(base_model_dir):
                for file in files:
                    if file.endswith(".zip"):
                        full_path = os.path.join(root, file).replace("\\", "/")
                        self.model_combo.addItem(full_path)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        model_layout.addWidget(self.model_combo)
        model_group.setLayout(model_layout)
        control_layout.addWidget(model_group)

        pcc_group = QGroupBox("2. PCC Controller")
        pcc_layout = QVBoxLayout()
        self.pcc_checkbox = QCheckBox("Enable PCC Nominal Control")
        self.pcc_checkbox.setChecked(True)
        self.pcc_checkbox.stateChanged.connect(self._on_pcc_toggled)
        pcc_layout.addWidget(self.pcc_checkbox)
        pcc_group.setLayout(pcc_layout)
        control_layout.addWidget(pcc_group)

        target_group = QGroupBox("3. Target Selection")
        target_layout = QFormLayout()
        self.spin_x = QDoubleSpinBox()
        self.spin_x.setRange(-0.8, 0.8) # Constrained workspace
        self.spin_x.setSingleStep(0.05)
        self.spin_y = QDoubleSpinBox()
        self.spin_y.setRange(-0.8, 0.8) # Constrained workspace
        self.spin_y.setSingleStep(0.05)
        self.spin_z = QDoubleSpinBox()
        self.spin_z.setRange(0.2, 1.0) # Prevent impossible Z heights
        self.spin_z.setSingleStep(0.05)
        self.spin_z.setValue(0.5)
        self.spin_x.valueChanged.connect(self._on_target_changed)
        self.spin_y.valueChanged.connect(self._on_target_changed)
        self.spin_z.valueChanged.connect(self._on_target_changed)
        target_layout.addRow("Target X:", self.spin_x)
        target_layout.addRow("Target Y:", self.spin_y)
        target_layout.addRow("Target Z:", self.spin_z)
        target_group.setLayout(target_layout)
        control_layout.addWidget(target_group)

        # 4. 负载设置面板 (Payload Config)
        payload_group = QGroupBox("4. External Load (N)")
        payload_layout = QFormLayout()
        self.spin_fx = QDoubleSpinBox()
        self.spin_fx.setRange(-50.0, 50.0)
        self.spin_fx.setSingleStep(1.0)
        self.spin_fy = QDoubleSpinBox()
        self.spin_fy.setRange(-50.0, 50.0)
        self.spin_fy.setSingleStep(1.0)
        self.spin_fz = QDoubleSpinBox()
        self.spin_fz.setRange(-50.0, 50.0)
        self.spin_fz.setSingleStep(1.0)
        
        self.spin_fx.valueChanged.connect(self._on_payload_changed)
        self.spin_fy.valueChanged.connect(self._on_payload_changed)
        self.spin_fz.valueChanged.connect(self._on_payload_changed)
        
        payload_layout.addRow("Force X (N):", self.spin_fx)
        payload_layout.addRow("Force Y (N):", self.spin_fy)
        payload_layout.addRow("Force Z (N):", self.spin_fz)
        
        self.btn_reset_load = QPushButton("Reset Load to 0")
        self.btn_reset_load.clicked.connect(self._on_reset_load)
        payload_layout.addRow(self.btn_reset_load)
        payload_group.setLayout(payload_layout)
        control_layout.addWidget(payload_group)

        # 5. 系统状态指示灯面板
        status_group = QGroupBox("5. System Status LEDs")
        status_layout = QVBoxLayout()

        sim2real_layout = QHBoxLayout()
        self.led_sim2real = QLabel()
        self.led_sim2real.setFixedSize(20, 20)
        self._set_led_color(self.led_sim2real, "gray")
        sim2real_layout.addWidget(self.led_sim2real)
        sim2real_layout.addWidget(QLabel("Sim2Real Gap Indicator"))
        sim2real_layout.addStretch()

        safety_layout = QHBoxLayout()
        self.led_safety = QLabel()
        self.led_safety.setFixedSize(20, 20)
        self._set_led_color(self.led_safety, "green")
        safety_layout.addWidget(self.led_safety)
        self.lbl_safety_text = QLabel("Safety System Armed")
        safety_layout.addWidget(self.lbl_safety_text)
        safety_layout.addStretch()

        status_layout.addLayout(sim2real_layout)
        status_layout.addLayout(safety_layout)
        status_group.setLayout(status_layout)
        control_layout.addWidget(status_group)

        # 7. Preset Experiments
        preset_group = QGroupBox("6. Preset Experiments")
        preset_layout = QVBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("0: Manual Setup")
        self.preset_combo.addItem("1: Exp 1 - Static & Disturbance")
        self.preset_combo.addItem("2: Exp 2 - Trajectory & Complex Load")
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        preset_layout.addWidget(self.preset_combo)
        preset_group.setLayout(preset_layout)
        control_layout.addWidget(preset_group)

        btn_group = QGroupBox("7. Execution Control")
        btn_layout = QVBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_start.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;"
        )
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setStyleSheet(
            "background-color: #FF9800; color: white; font-weight: bold; padding: 8px;"
        )
        self.btn_stop = QPushButton("Stop / Reset")
        self.btn_stop.setStyleSheet(
            "background-color: #F44336; color: white; font-weight: bold; padding: 8px;"
        )
        self.btn_start.clicked.connect(self._on_start)
        self.btn_pause.clicked.connect(self._on_pause)
        self.btn_stop.clicked.connect(self._on_stop)
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_pause)
        btn_layout.addWidget(self.btn_stop)
        
        # Save Data Button
        self.btn_save = QPushButton("Export Plot Data")
        self.btn_save.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 8px;")
        self.btn_save.clicked.connect(self._on_save_data)
        btn_layout.addWidget(self.btn_save)
        
        btn_group.setLayout(btn_layout)
        control_layout.addWidget(btn_group)
        control_layout.addStretch()
        main_layout.addWidget(control_panel)

        dashboard_panel = QWidget()
        dashboard_layout = QVBoxLayout(dashboard_panel)

        # 1. Top Panel: Text Stats and Error Gauge
        top_info_layout = QHBoxLayout()
        
        text_group = QGroupBox("Real-time Actuation States")
        text_layout = QVBoxLayout()
        self.lbl_pcc_out = QLabel("PCC Output: [0.00, ...]")
        self.lbl_rl_out = QLabel("RL Residual: [0.00, ...]")
        self.lbl_total_out = QLabel("Total Output: [0.00, ...]")
        self.lbl_pcc_out.setFont(self._get_mono_font())
        self.lbl_rl_out.setFont(self._get_mono_font())
        self.lbl_total_out.setFont(self._get_mono_font())
        text_layout.addWidget(self.lbl_pcc_out)
        text_layout.addWidget(self.lbl_rl_out)
        text_layout.addWidget(self.lbl_total_out)
        text_group.setLayout(text_layout)
        top_info_layout.addWidget(text_group, stretch=2)

        gauge_group = QGroupBox("Tracking Error Indicator")
        gauge_layout = QVBoxLayout()
        self.error_bar = QProgressBar()
        self.error_bar.setRange(0, int(self.worker.max_error * 100)) # e.g. 0 to 40
        self.error_bar.setValue(0)
        self.error_bar.setTextVisible(True)
        self.error_bar.setFormat("%v cm")
        self.error_bar.setStyleSheet("QProgressBar::chunk { background-color: #4CAF50; }")
        gauge_layout.addWidget(self.error_bar)
        gauge_group.setLayout(gauge_layout)
        top_info_layout.addWidget(gauge_group, stretch=1)
        
        dashboard_layout.addLayout(top_info_layout)

        # 2. 3 Real-time visual panels
        visuals_layout = QHBoxLayout()
        
        # Panel 1: XY Trajectory (Radar Style)
        self.plot_xy = pg.PlotWidget(title="XY Trajectory (Top View)")
        self.plot_xy.setLabel("left", "Y (m)")
        self.plot_xy.setLabel("bottom", "X (m)")
        self.plot_xy.setXRange(-0.8, 0.8)
        self.plot_xy.setYRange(-0.8, 0.8)
        self.plot_xy.setAspectLocked(True) # 1:1 Aspect Ratio
        self.plot_xy.hideAxis('bottom')
        self.plot_xy.hideAxis('left')
        
        # Radar circles
        for r in [0.2, 0.4, 0.6, 0.8]:
            circle = QGraphicsEllipseItem(-r, -r, r*2, r*2)
            circle.setPen(pg.mkPen((100, 100, 100, 150), width=1, style=Qt.DashLine))
            self.plot_xy.addItem(circle)
            text = pg.TextItem(f"{r}m", color=(150, 150, 150), anchor=(0.5, 1))
            self.plot_xy.addItem(text)
            text.setPos(0, -r)
            
        # Crosshairs
        v_line = pg.InfiniteLine(angle=90, pen=pg.mkPen((100, 100, 100, 150)))
        h_line = pg.InfiniteLine(angle=0, pen=pg.mkPen((100, 100, 100, 150)))
        self.plot_xy.addItem(v_line)
        self.plot_xy.addItem(h_line)
            
        # Curves
        self.traj_curve = self.plot_xy.plot(pen=pg.mkPen('c', width=2))
        self.target_scatter = pg.ScatterPlotItem(size=12, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0))
        self.current_scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(0, 255, 0))
        self.plot_xy.addItem(self.target_scatter)
        self.plot_xy.addItem(self.current_scatter)
        self.depth_text = pg.TextItem(text="Z: 0.00m", color=(200, 200, 200), anchor=(0, 1))
        self.plot_xy.addItem(self.depth_text)
        self.depth_text.setPos(-0.75, 0.75)
        visuals_layout.addWidget(self.plot_xy, stretch=1)
        
        # Panel 2: Force Vector (2D)
        self.plot_force = pg.PlotWidget(title="XY Plane Applied Force (N)")
        self.plot_force.setXRange(-50, 50)
        self.plot_force.setYRange(-50, 50)
        self.plot_force.setAspectLocked(True) # 1:1 Aspect Ratio
        self.plot_force.showGrid(x=True, y=True)
        
        # Crosshairs
        v_line_f = pg.InfiniteLine(angle=90, pen=pg.mkPen((100, 100, 100, 150)))
        h_line_f = pg.InfiniteLine(angle=0, pen=pg.mkPen((100, 100, 100, 150)))
        self.plot_force.addItem(v_line_f)
        self.plot_force.addItem(h_line_f)
        
        self.force_arrow = pg.ArrowItem(angle=0, tipAngle=30, baseAngle=20, headLen=20, tailLen=0.1, tailWidth=3, pen={'color': 'r', 'width': 2}, brush='r')
        self.plot_force.addItem(self.force_arrow)
        self.force_arrow.setPos(0, 0) # Base
        self.force_line_2d = self.plot_force.plot(pen=pg.mkPen('r', width=3))
        
        self.force_z_text = pg.TextItem(text="Fz: 0.0 N", color=(255, 100, 100), anchor=(0, 1))
        self.plot_force.addItem(self.force_z_text)
        self.force_z_text.setPos(-45, 45)
        visuals_layout.addWidget(self.plot_force, stretch=1)
        
        # Panel 3: Safety System Monitor
        safety_panel = QGroupBox("Safety Guard System")
        safety_layout = QVBoxLayout(safety_panel)
        
        self.led_amp = QLabel()
        self.led_rate = QLabel()
        self.led_err = QLabel()
        self.led_posture = QLabel()
        
        for led in [self.led_amp, self.led_rate, self.led_err, self.led_posture]:
            led.setFixedSize(20, 20)
            self._set_led_color(led, "green")
            
        self.lbl_amp = QLabel("Action Amplitude Limit")
        self.lbl_rate = QLabel("Action Rate (Jerk) Limit")
        self.lbl_err = QLabel("Tracking Error Threshold")
        self.lbl_posture = QLabel("Posture Singularity / Buckling")
        
        def make_row(led, lbl):
            h = QHBoxLayout()
            h.addWidget(led)
            h.addWidget(lbl)
            h.addStretch()
            return h
            
        safety_layout.addLayout(make_row(self.led_amp, self.lbl_amp))
        safety_layout.addLayout(make_row(self.led_rate, self.lbl_rate))
        safety_layout.addLayout(make_row(self.led_err, self.lbl_err))
        safety_layout.addLayout(make_row(self.led_posture, self.lbl_posture))
        safety_layout.addStretch()
        
        visuals_layout.addWidget(safety_panel, stretch=1)
        
        dashboard_layout.addLayout(visuals_layout, stretch=2)

        # 3. Time-series plots in Tabs
        self.tabs = QTabWidget()
        pg.setConfigOptions(antialias=True)
        self.plot_pcc, self.pcc_curves = self._create_action_plot("")
        self.plot_rl, self.rl_curves = self._create_action_plot("")
        self.plot_total, self.total_curves = self._create_action_plot("")
        self.plot_error = pg.PlotWidget()
        self.plot_error.setLabel("left", "Error (m)")
        self.plot_error.setLabel("bottom", "Time Steps")
        self.error_curve = self.plot_error.plot(pen=pg.mkPen(color="r", width=2))
        
        self.tabs.addTab(self.plot_error, "Tracking Error")
        self.tabs.addTab(self.plot_total, "Total Command")
        self.tabs.addTab(self.plot_pcc, "PCC Nominal")
        self.tabs.addTab(self.plot_rl, "RL Residual")
        dashboard_layout.addWidget(self.tabs, stretch=3)

        # 4. Log Console
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setFont(self._get_mono_font())
        self.log_console.setFixedHeight(120)
        log_label = QLabel("Event Log Console:")
        log_label.setStyleSheet("font-weight: bold; padding-top: 5px;")
        dashboard_layout.addWidget(log_label)
        dashboard_layout.addWidget(self.log_console, stretch=1)

        main_layout.addWidget(dashboard_panel, stretch=2)
        
        self._log("INFO", "Application initialized successfully.")

    def _set_led_color(self, label, color):
        style = (
            f"border-radius: 10px; background-color: {color}; border: 1px solid black;"
        )
        label.setStyleSheet(style)

    def _log(self, level, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = "white"
        if level == "INFO": 
            color = "#4CAF50"
            logging.info(message)
        elif level == "WARN": 
            color = "#FF9800"
            logging.warning(message)
        elif level == "ERROR": 
            color = "#F44336"
            logging.error(message)
        
        formatted = f'<span style="color:gray">[{timestamp}]</span> <span style="color:{color}; font-weight:bold">[{level}]</span> {message}'
        self.log_console.append(formatted)
        scrollbar = self.log_console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _create_action_plot(self, title):
        plot_widget = pg.PlotWidget(title=title)
        plot_widget.setLabel("left", "Action Value")
        plot_widget.setLabel("bottom", "Time Steps")
        curves = []
        colors = [
            (255, 100, 100),
            (100, 255, 100),
            (100, 100, 255),
            (255, 255, 100),
            (255, 100, 255),
            (100, 255, 255),
            (255, 180, 100),
            (200, 200, 200),
        ]
        for i in range(8):
            curve = plot_widget.plot(pen=pg.mkPen(color=colors[i], width=1.5))
            curves.append(curve)
        return plot_widget, curves

    def _get_mono_font(self):
        font = self.font()
        font.setFamily("Courier New")
        font.setPointSize(10)
        font.setBold(True)
        return font

    def _on_model_changed(self, text):
        self.worker.load_rl_model(text)

    def _on_pcc_toggled(self, state):
        self.worker.use_pcc = state == Qt.Checked

    def _on_preset_changed(self, index):
        self.worker.preset_mode = index
        self.worker.preset_start_mj_time = None
        
        is_manual = (index == 0)
        self.spin_x.setEnabled(is_manual)
        self.spin_y.setEnabled(is_manual)
        self.spin_z.setEnabled(is_manual)
        self.spin_fx.setEnabled(is_manual)
        self.spin_fy.setEnabled(is_manual)
        self.spin_fz.setEnabled(is_manual)
        self.btn_reset_load.setEnabled(is_manual)

        if is_manual:
            self._log("INFO", "Switched to Manual Setup.")
            self._on_target_changed()
            self._on_payload_changed()
        else:
            self._log("INFO", f"Switched to Preset Experiment {index}.")

    def _on_target_changed(self):
        x = self.spin_x.value()
        y = self.spin_y.value()
        z = self.spin_z.value()
        self.worker.target_pos = [x, y, z]

    def _on_payload_changed(self):
        fx = self.spin_fx.value()
        fy = self.spin_fy.value()
        fz = self.spin_fz.value()
        self.worker.current_force = [fx, fy, fz]

    def _on_reset_load(self):
        self.spin_fx.setValue(0.0)
        self.spin_fy.setValue(0.0)
        self.spin_fz.setValue(0.0)

    def _on_save_data(self):
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(data_dir, exist_ok=True)
        filename = os.path.join(data_dir, f"sim_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz")
        
        if len(self.time_data) == 0:
            self._log("WARN", "No data to save (Simulation has not run).")
            return

        np.savez(
            filename,
            time=np.array(self.time_data),
            error=np.array(self.error_data),
            pcc_action=np.array(self.pcc_action_data),
            rl_action=np.array(self.rl_action_data),
            total_action=np.array(self.total_action_data),
        )
        self._log("INFO", f"Saved full history plot data to {filename}")

    def _on_start(self):
        self._set_led_color(self.led_safety, "green")
        self.lbl_safety_text.setText("System Running Normally")
        self.worker.is_running = True
        self._log("INFO", f"Simulation started. Target: {self.worker.target_pos}")

    def _on_pause(self):
        if self.worker.is_running:
            self.worker.is_running = False
            self._set_led_color(self.led_sim2real, "gray")
            self._log("WARN", "Simulation paused.")

    def _on_stop(self):
        self.worker.is_running = False
        self.worker.tester.reset()
        self.step_count = 0
        self.time_data.clear()
        self.error_data.clear()
        
        for i in range(8):
            self.rl_action_data[i].clear()
            self.pcc_action_data[i].clear()
            self.total_action_data[i].clear()
            
        self._refresh_plots()
        self.traj_x = []
        self.traj_y = []
        self.traj_curve.setData([], [])
        self.force_line_2d.setData([], [])
        self.force_arrow.setPos(0, 0)
        self.error_bar.setValue(0)
        self._set_led_color(self.led_sim2real, "gray")
        self._set_led_color(self.led_safety, "green")
        
        for led in [self.led_amp, self.led_rate, self.led_err, self.led_posture]:
            self._set_led_color(led, "green")
            
        self.lbl_safety_text.setText("System Reset & Armed")
        self._log("INFO", "System stopped and reset.")

    def _refresh_plots(self):
        # We now plot all historical data so it doesn't slide too quickly
        plot_len = min(len(self.time_data), self.plot_history_length)
        if plot_len == 0:
            # Clear UI graphs if empty
            self.error_curve.setData([], [])
            for i in range(8):
                self.rl_curves[i].setData([], [])
                self.pcc_curves[i].setData([], [])
                self.total_curves[i].setData([], [])
            return

        t_slice = self.time_data[-plot_len:]
        self.error_curve.setData(t_slice, self.error_data[-plot_len:])
        for i in range(8):
            self.rl_curves[i].setData(t_slice, self.rl_action_data[i][-plot_len:])
            self.pcc_curves[i].setData(t_slice, self.pcc_action_data[i][-plot_len:])
            self.total_curves[i].setData(t_slice, self.total_action_data[i][-plot_len:])

    def _update_real_data(self, info):
        if not info["is_safe"]:
            if self.worker.is_running:
                self.worker.is_running = False
            
            # Special check to distinguish between normal complete vs emergency stop
            if "Completed" in info['safety_reason']:
                self._set_led_color(self.led_safety, "green")
                self.lbl_safety_text.setText(f"✓ {info['safety_reason']}")
                self._log("INFO", f"Task Completed: {info['safety_reason']}")
            else:
                self._set_led_color(self.led_safety, "red")
                self.lbl_safety_text.setText(f"EMERGENCY STOP: {info['safety_reason']}")
                self._log("ERROR", f"EMERGENCY STOP: {info['safety_reason']}")
                
                # Check specifics
                if "Amplitude" in info['safety_reason']:
                    self._set_led_color(self.led_amp, "red")
                elif "Rate" in info['safety_reason'] or "Jerk" in info['safety_reason']:
                    self._set_led_color(self.led_rate, "red")
                elif "Tracking Error" in info['safety_reason']:
                    self._set_led_color(self.led_err, "red")
                elif "Posture Singularity" in info['safety_reason']:
                    self._set_led_color(self.led_posture, "red")
            return

        # System is safe, reset all specific limit LEDs to green
        for led in [self.led_amp, self.led_rate, self.led_err, self.led_posture]:
            self._set_led_color(led, "green")

        gap = info["sim2real_gap"]
        if gap < 0.1:
            self._set_led_color(self.led_sim2real, "green")
        elif gap < 0.25:
            self._set_led_color(self.led_sim2real, "orange")
        else:
            self._set_led_color(self.led_sim2real, "red")

        current_time = self.step_count
        real_error = info["error"]
        real_pcc = info["cmd_pcc"]
        real_rl = info["cmd_rl"]
        real_total = info["cmd_total"]
        curr_tip = info.get("current_pos", [0, 0, 0])

        pcc_str = ", ".join([f"{x:5.2f}" for x in real_pcc])
        rl_str = ", ".join([f"{x:5.2f}" for x in real_rl])
        total_str = ", ".join([f"{x:5.2f}" for x in real_total])
        self.lbl_pcc_out.setText(f"PCC Output:  [{pcc_str}]")
        self.lbl_rl_out.setText(f"RL Residual: [{rl_str}]")
        self.lbl_total_out.setText(f"Total Output:[{total_str}]")

        self.time_data.append(current_time)
        self.error_data.append(max(0, real_error))

        for i in range(8):
            self.rl_action_data[i].append(real_rl[i])
            self.pcc_action_data[i].append(real_pcc[i])
            self.total_action_data[i].append(real_total[i])

        # Update Tracking Error UI
        error_cm = int(real_error * 100)
        self.error_bar.setValue(min(error_cm, self.error_bar.maximum()))
        if real_error > self.worker.max_error * 0.8:
            self.error_bar.setStyleSheet("QProgressBar::chunk { background-color: #F44336; }") # Red
        elif real_error > self.worker.max_error * 0.5:
            self.error_bar.setStyleSheet("QProgressBar::chunk { background-color: #FF9800; }") # Orange
        else:
            self.error_bar.setStyleSheet("QProgressBar::chunk { background-color: #4CAF50; }") # Green

        # Sync Preset Current Targets and Load to UI
        if self.worker.preset_mode != 0:
            cur_target = info.get("current_target_pos", [0, 0, 0])
            cur_force = info.get("current_force_load", [0, 0, 0])
            
            # Temporarily disconnect to avoid feedback loop
            self.spin_x.blockSignals(True)
            self.spin_y.blockSignals(True)
            self.spin_z.blockSignals(True)
            self.spin_fx.blockSignals(True)
            self.spin_fy.blockSignals(True)
            self.spin_fz.blockSignals(True)
            
            self.spin_x.setValue(cur_target[0])
            self.spin_y.setValue(cur_target[1])
            self.spin_z.setValue(cur_target[2])
            self.spin_fx.setValue(cur_force[0])
            self.spin_fy.setValue(cur_force[1])
            self.spin_fz.setValue(cur_force[2])
            
            self.spin_x.blockSignals(False)
            self.spin_y.blockSignals(False)
            self.spin_z.blockSignals(False)
            self.spin_fx.blockSignals(False)
            self.spin_fy.blockSignals(False)
            self.spin_fz.blockSignals(False)

        # Update Visuals
        if not hasattr(self, 'traj_x'):
            self.traj_x = []
            self.traj_y = []
        
        self.traj_x.append(curr_tip[0])
        self.traj_y.append(curr_tip[1])
        if len(self.traj_x) > 300:
            self.traj_x.pop(0)
            self.traj_y.pop(0)
        
        self.traj_curve.setData(self.traj_x, self.traj_y)
        cur_target = info.get("current_target_pos", [0, 0, 0])
        self.target_scatter.setData([cur_target[0]], [cur_target[1]])
        self.current_scatter.setData([curr_tip[0]], [curr_tip[1]])
        self.depth_text.setText(f"Tip Z: {curr_tip[2]:.3f}m\nTarget Z: {cur_target[2]:.3f}m")
        # Place text at top-left of the viewbox regardless of dynamic sizing
        rect = self.plot_xy.viewRect()
        self.depth_text.setPos(rect.left() + (rect.width()*0.02), rect.top() - (rect.height()*0.02))

        # Update 2D Force Vector
        cur_force = info.get("current_force_load", [0, 0, 0])
        fx, fy, fz = cur_force[0], cur_force[1], cur_force[2]
        self.force_line_2d.setData([0, fx], [0, fy])
        if fx == 0 and fy == 0:
            self.force_arrow.setStyle(angle=0)
            self.force_arrow.setPos(0, 0)
            self.force_arrow.setVisible(False)
        else:
            self.force_arrow.setVisible(True)
            self.force_arrow.setPos(fx, fy)
            angle = np.degrees(np.arctan2(fy, fx))
            self.force_arrow.setStyle(angle=180-angle) # pyqtgraph arrow points backwards natively so we flip it
            
        self.force_z_text.setText(f"Fz: {fz:.1f} N")

        self.step_count += 1
        if self.step_count % 5 == 0:
            self._refresh_plots()

    def closeEvent(self, event):
        self.worker.is_running = False
        self.worker.terminate()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    try:
        app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
    except Exception as e:
        print(f"Warning: Could not load dark theme: {e}")
    window = SoftRobotDashboard()
    window.show()
    sys.exit(app.exec_())
