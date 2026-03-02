import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.core.tester import SoftRobotTester
from src.core.fuzzy_controller import FuzzyAlphaController

def test_fuzzy_tracking():
    print("=== Testing Soft Robot Control with Fuzzy Alpha Scaling ===")
    
    xml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "env", "assets", "one_disks_uj.xml"))
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "ppo_soft_robot.zip"))
    
    # We will initialize the tester in 'hybrid' mode but manually control alpha
    tester = SoftRobotTester(xml_path=xml_path, model_path=model_path, mode="hybrid", render=True)
    
    # Check if the RL model actually loaded.
    if not tester.model_loaded:
        print("[!] Warning: RL Model not found or loaded. The test will run, but residuals will be zero.")
    
    fuzzy_controller = FuzzyAlphaController(dt=tester.mj_model.opt.timestep, lpf_gamma=0.2)
    
    # Base scale parameter from RL 
    base_residual_scale = 0.08
    tester.residual_scale = base_residual_scale # Reset just in case
    
    # Define a trajectory (a square/circle)
    t = np.linspace(0, 2 * np.pi, 600)
    radius = 0.15
    center = np.array([0.0, 0.0, 0.9])
    
    # Arrays for plotting
    time_history = []
    error_history = []
    alpha_history = []
    
    print("Pre-stabilizing...")
    tester.stabilize_at(center, tolerance=0.01)
    
    print("\nStarting trajectory tracking with Fuzzy Alpha...")
    start_time = time.time()
    
    try:
        for idx in range(len(t)):
            # Update target position
            target_x = center[0] + radius * np.cos(t[idx])
            target_y = center[1] + radius * np.sin(t[idx])
            target_z = center[2]
            
            current_target = np.array([target_x, target_y, target_z])
            tester.set_target(current_target)
            
            # Step physics
            info = tester.step()
            
            # Use fuzzy controller to compute scaling coefficient for NEXT step
            current_error = info["error"]
            alpha = fuzzy_controller.step(current_error)
            
            # Dynamic scaling of the RL residual action
            tester.residual_scale = base_residual_scale * alpha
            
            # Logging
            time_history.append(info["time"])
            error_history.append(current_error)
            alpha_history.append(alpha)
            
            # Print brief status
            if idx % 50 == 0:
                print(f"Step {idx:3d} | Err: {current_error:.4f}m | dtError: {fuzzy_controller.filtered_error_dot:+.3f}m/s | Alpha Cfg: {alpha:.3f} | Resid Scl: {tester.residual_scale:.4f}")
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    finally:
        tester.close()
        
    print(f"\nSimulation Finished in {time.time() - start_time:.2f}s")
    
    # --- Plotting the Results ---
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    color1 = 'tab:red'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Tracking Error (m)', color=color1)
    ax1.plot(time_history, error_history, color=color1, label='Error')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()  
    color2 = 'tab:blue'
    ax2.set_ylabel('Fuzzy Alpha Scaler', color=color2)  
    ax2.plot(time_history, alpha_history, color=color2, label='Alpha', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0.0, 1.1)
    
    fig.tight_layout()  
    plt.title('Tracking Error and Dynamic Fuzzy Alpha Scaling over Time')
    plt.savefig('fuzzy_tracking_results.png', dpi=300)
    print("Saved tracking plot to 'fuzzy_tracking_results.png'")
    # plt.show() # Uncomment to show plot immediately

if __name__ == "__main__":
    test_fuzzy_tracking()
