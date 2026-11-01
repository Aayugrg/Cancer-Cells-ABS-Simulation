import time
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import vispy.io as io
from vispy.app import Timer, run, process_events

from config import *
from vector_engine import VectorSimulationNumba
from renderer import SimulationRenderer

# =============================================================================
# SNAPSHOT CONFIGURATION
# =============================================================================
SNAPSHOT_INTERVAL_HOURS = 24.0  # Export snapshots every K hours
SLICE_Z_TARGET = 75             # Z-plane coordinate for 2D slice
SAVE_SNAPSHOTS = True
SNAPSHOT_DIR = "simulation_snapshots_live"

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

# =============================================================================
# SIMULATION INITIALIZATION
# =============================================================================
sim = VectorSimulationNumba()
sim.seed(NUM_A_INIT, NUM_B_INIT, NUM_C_INIT, NUM_D_INIT, NUM_E_INIT)

renderer = SimulationRenderer(title="Live Parallel Engine - Volume Telemetry & Snapshots")

TOTAL_STEPS = TOTAL_DAYS * STEPS_PER_DAY
current_step = 0

def capture_2d_slice(sim, z_target, hour_label, day_label, c_vol, t_vol):
    """Extracts a 2D cross-sectional slice with embedded Legend and Volume metrics."""
    slice_tensor = sim.State[z_target, :, :, :]
    occupied_mask = slice_tensor[:, :, OCCUPIED] == 1.0

    if not occupied_mask.any():
        print(f" 2D Slice at Z={z_target} is empty at Hour {hour_label:.1f}.")
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
    plt.xlabel("X Coordinate (Voxels)", fontsize=8)
    plt.ylabel("Y Coordinate (Voxels)", fontsize=8)
    plt.grid(True, linestyle=':', alpha=0.25)
    plt.tight_layout()

    filename = os.path.join(SNAPSHOT_DIR, f"live_2d_z{z_target}_hour_{int(hour_label):04d}.png")
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"  Saved 2D Slice -> {filename}")

def update_live_frame(event):
    global current_step

    if current_step >= TOTAL_STEPS:
        print("Simulation Complete!")
        timer.stop()
        return

    t0 = time.perf_counter()
    sim.step()
    dt = time.perf_counter() - t0

    current_step += 1
    current_hour = current_step * (TIME_STEP_MIN / 60.0)
    current_day = current_step / STEPS_PER_DAY

    # 1. Extract Cell Counts
    a_cnt, b_cnt, c_cnt, d_n, d_a, d_e, e_cnt = sim.get_counts()
    d_tot = d_n + d_a + d_e

    # 2. Compute Tumor Volumes at EVERY step
    c_vol_mm3, total_vol_mm3 = calculate_tumor_volumes(a_cnt, b_cnt, c_cnt, d_tot, e_cnt)

    # 3. Update Visuals & HUD
    positions, colors = sim.extract_render_buffers()
    renderer.update_visuals(positions, colors, current_day, TOTAL_DAYS, c_vol_mm3, total_vol_mm3)

    # 4. Snapshot Export
    if SAVE_SNAPSHOTS and (current_step % SNAPSHOT_STEP_INTERVAL == 0):
        img_3d = renderer.canvas.render()
        fn_3d = os.path.join(SNAPSHOT_DIR, f"live_3d_hour_{int(current_hour):04d}.png")
        io.write_png(fn_3d, img_3d)
        print(f" [Hour {current_hour:.0f} / Day {current_day:.2f}] Saved 3D Snapshot -> {fn_3d}")
        capture_2d_slice(sim, SLICE_Z_TARGET, current_hour, current_day, c_vol_mm3, total_vol_mm3)

    # Console Telemetry
    print(f"Step {current_step:4d}/{TOTAL_STEPS} | Hour {current_hour:5.1f}h | Cancer Vol: {c_vol_mm3:.4f} mm³ | Total Vol: {total_vol_mm3:.4f} mm³ | DT: {dt*1000:5.1f} ms")
    process_events()

timer = Timer(interval=0.01, connect=update_live_frame, start=True)

if __name__ == "__main__":
    print("Launching Live Simulation ")
    run()