import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import os

def visualize_sim_data(file_path):
    print(f"Loading data from: {file_path}")
    data = np.load(file_path)
    
    time = data['time']
    error = data['error']
    pcc_action = data['pcc_action']
    rl_action = data['rl_action']
    total_action = data['total_action']
    
    # Check if there is actual data to plot
    if len(time) == 0:
        print("Error: The data array is empty.")
        return

    # Set up styling
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
    fig.canvas.manager.set_window_title("Sim2Real Offline Data Visualization")
    
    colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:orange', 
              'tab:purple', 'tab:cyan', 'tab:pink', 'tab:gray']

    # 1. Plot Tracking Error
    axes[0].plot(time, error, color='red', linewidth=2.5)
    axes[0].set_title('End-Effector Tracking Error (Euclidean Distance)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Error (m)', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].fill_between(time, error, alpha=0.1, color='red')

    # 2. Plot PCC Actions (8 dimensions)
    for i in range(8):
        axes[1].plot(time, pcc_action[i], color=colors[i], alpha=0.8, linewidth=1.5, label=f'Act {i+1}' if i < 4 else "")
    axes[1].set_title('PCC Nominal Baseline Commands', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Length (m)', fontsize=12)
    axes[1].legend(loc='upper right', ncol=4, fontsize=9)

    # 3. Plot RL Residual Actions
    for i in range(8):
        axes[2].plot(time, rl_action[i], color=colors[i], alpha=0.8, linewidth=1.5)
    axes[2].set_title('RL Residual Corrections', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Correction (m)', fontsize=12)

    # 4. Plot Total Actions
    for i in range(8):
        axes[3].plot(time, total_action[i], color=colors[i], alpha=0.8, linewidth=1.5)
    axes[3].set_title('Total Synthesized Commands (PCC + RL)', fontsize=12, fontweight='bold')
    axes[3].set_ylabel('Total Length (m)', fontsize=12)
    axes[3].set_xlabel('Simulation Time Steps', fontsize=12, fontweight='bold')

    # Formatting and Layout
    plt.tight_layout()
    
    # Save a static image version
    pic_name = file_path.replace('.npz', '.png')
    plt.savefig(pic_name, dpi=300)
    print(f"Plot saved to: {pic_name}")
    
    # Show the interactive window
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        # If no file provided, find the newest .npz file in the data folder
        data_files = glob.glob('src/sim2real/data/*.npz')
        if not data_files:
            print("No .npz files found in src/sim2real/data/ !")
            sys.exit(1)
        target_file = max(data_files, key=os.path.getctime)
        
    visualize_sim_data(target_file)
