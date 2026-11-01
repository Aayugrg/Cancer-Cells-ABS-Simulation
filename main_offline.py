import time
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from vispy.app import Timer, run
from config import *
from vector_engine import VectorSimulationNumba
from renderer import SimulationRenderer

# =============================================================================
# CONFIGURATION SETTINGS
# =============================================================================
SNAPSHOT_INTERVAL_HOURS = 24.0  # Generate 2D slices every K hours
SLICE_Z_TARGET = 75             # Z-plane for 2D slice
SAVE_SNAPSHOTS = True
SNAPSHOT_DIR = "simulation_snapshots_offline"
CACHE_FILE = "tumor_trajectory_40days.npz"

if SAVE_SNAPSHOTS and not os.path.exists(SNAPSHOT_DIR):
    os.makedirs(SNAPSHOT_DIR)

STEPS_PER_HOUR = 60.0 / TIME_STEP_MIN
SNAPSHOT_STEP_INTERVAL = int(SNAPSHOT_INTERVAL_HOURS * STEPS_PER_HOUR)

def calculate_tumor_volumes(count_a, count_b, count_c, count_d, count_e):
    """
    Computes cancer-only and total cellular volumes in mm³ 
    based on physical cell radii (Cancer: r=10µm, Microenvironment: r=5µm).
    """
    pi_val = math.pi
    vol_cancer_voxels = (count_a + count_b) * ((4.0 / 3.0) * pi_val * (10.0**3))
    vol_other_voxels = (count_c + count_d + count_e) * ((4.0 / 3.0) * pi_val * (5.0**3))
    
    cancer_vol_mm3 = vol_cancer_voxels / (10.0**9)
    total_vol_mm3 = (vol_cancer_voxels + vol_other_voxels) / (10.0**9)
    return cancer_vol_mm3, total_vol_mm3

def capture_2d_slice_offline(sim, z_target, hour_label, day_label, c_vol, t_vol):
    """Generates 2D slice snapshot with legend and volumes during pre-computation."""
    slice_tensor = sim.State[z_target, :, :, :]
    occupied_mask = slice_tensor[:, :, OCCUPIED] == 1.0

    if not occupied_mask.any():
        return

    y_coords, x_coords = np.argwhere(occupied_mask)[:, 0], np.argwhere(occupied_mask)[:, 1]
    cell_types = slice_tensor[y_coords, x_coords, CELL_TYPE].astype(int)
    sub_states = slice_tensor[y_coords, x_coords, SUB_STATE].astype(int)

    color_list = []
    for i in range(len(cell_types)):
        t = cell_types[i]
        key = 40 + sub_states[i] if t == 4 else t
        rgba = COLOR_MAP.get(key, [1.0, 1.0, 1.0, 0.8])
        color_list.append(rgba[:3])

    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=200)
    plt.style.use('dark_background')
    
    ax.scatter(x_coords, y_coords, c=color_list, s=14, alpha=0.9, marker='o')
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_aspect('equal', adjustable='box')

    legend_patches = [
        mpatches.Patch(color='#FF3333', label='Agent A (CXCR4-ve)'),
        mpatches.Patch(color='#FF8000', label='Agent B (CXCR4+ve)'),
        mpatches.Patch(color='#33CC33', label='Agent C (Stromal)'),
        mpatches.Patch(color='#3399FF', label='Agent D (Naive)'),
        mpatches.Patch(color='#00FFFF', label='Agent D (Activated)'),
        mpatches.Patch(color='#9933FF', label='Agent D (Exhausted)'),
        mpatches.Patch(color='#FFFF00', label='Agent E (Treg)')
    ]
    ax.legend(handles=legend_patches, loc='upper left', fontsize=7, framealpha=0.6, facecolor='#111111')

    plt.title(f"2D Midline Cross-Section (Z = {z_target})\nDay {day_label:.2f} ({hour_label:.0f}h) | Cancer Vol: {c_vol:.4f} mm³ | Total Vol: {t_vol:.4f} mm³", 
              fontsize=9, fontweight='bold', pad=10)
    plt.xlabel("X Position (Voxels)", fontsize=8)
    plt.ylabel("Y Position (Voxels)", fontsize=8)
    plt.grid(True, linestyle=':', alpha=0.25)
    plt.tight_layout()

    filename = os.path.join(SNAPSHOT_DIR, f"offline_2d_z{z_target}_hour_{int(hour_label):04d}.png")
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"  📸 Saved 2D Slice -> {filename}")

def generate_and_cache_trajectory():
    print("=========================================================================================")
    print(f"       PRE-COMPUTING {TOTAL_DAYS}-DAY TRAJECTORY ({TOTAL_STEPS} STEPS)")
    print("=========================================================================================")

    sim = VectorSimulationNumba()
    sim.seed(NUM_A_INIT, NUM_B_INIT, NUM_C_INIT, NUM_D_INIT, NUM_E_INIT)

    positions_list, colors_list, days_list, cancer_vols_list, total_vols_list = [], [], [], [], []
    t0_all = time.perf_counter()

    for step in range(1, TOTAL_STEPS + 1):
        t0 = time.perf_counter()
        sim.step()
        dt = time.perf_counter() - t0

        current_hour = step * (TIME_STEP_MIN / 60.0)
        current_day = step / STEPS_PER_DAY

        a_cnt, b_cnt, c_cnt, d_n, d_a, d_e, e_cnt = sim.get_counts()
        d_tot = d_n + d_a + d_e

        # Calculate volumes at EVERY step
        c_vol_mm3, total_vol_mm3 = calculate_tumor_volumes(a_cnt, b_cnt, c_cnt, d_tot, e_cnt)

        pos, cols = sim.extract_render_buffers()

        positions_list.append(pos)
        colors_list.append(cols)
        days_list.append(current_day)
        cancer_vols_list.append(c_vol_mm3)
        total_vols_list.append(total_vol_mm3)

        if SAVE_SNAPSHOTS and (step % SNAPSHOT_STEP_INTERVAL == 0):
            capture_2d_slice_offline(sim, SLICE_Z_TARGET, current_hour, current_day, c_vol_mm3, total_vol_mm3)

        if step % STEPS_PER_DAY == 0:
            print(f" >>> DAY {int(current_day):2d}/{TOTAL_DAYS} | Cancer Vol: {c_vol_mm3:.4f} mm³ | Total Vol: {total_vol_mm3:.4f} mm³ | DT: {dt*1000:.1f} ms <<<")

    print(f"\nSimulation complete in {time.perf_counter()-t0_all:.2f} seconds. Caching trajectory...")

    np.savez_compressed(
        CACHE_FILE,
        days=np.array(days_list),
        cancer_vols=np.array(cancer_vols_list),
        total_vols=np.array(total_vols_list),
        num_frames=len(positions_list),
        **{f"pos_{i}": positions_list[i] for i in range(len(positions_list))},
        **{f"col_{i}": colors_list[i] for i in range(len(colors_list))}
    )
    print(f"Successfully cached dataset: '{CACHE_FILE}'\n")

def run_offline_replay():
    if not os.path.exists(CACHE_FILE):
        generate_and_cache_trajectory()

    print("Loading pre-computed trajectory data...")
    data = np.load(CACHE_FILE)

    num_frames = int(data['num_frames'])
    days = data['days']
    c_vols = data['cancer_vols']
    t_vols = data['total_vols']

    renderer = SimulationRenderer(title="Offline Mode: 60 FPS Replay with Volume HUD")
    frame_idx = 0

    def replay_step(event):
        nonlocal frame_idx
        if frame_idx >= num_frames:
            frame_idx = 0

        pos = data[f"pos_{frame_idx}"]
        col = data[f"col_{frame_idx}"]
        day_num = days[frame_idx]
        c_vol = c_vols[frame_idx]
        t_vol = t_vols[frame_idx]

        renderer.update_visuals(pos, col, day_num, TOTAL_DAYS, c_vol, t_vol)
        frame_idx += 1

    timer = Timer(interval=1/60.0, connect=replay_step, start=True)
    print("Launching 60 FPS Interactive Playback Window")
    run()

if __name__ == "__main__":
    run_offline_replay()