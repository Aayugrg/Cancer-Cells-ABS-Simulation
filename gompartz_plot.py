import time
import math
import numpy as np
import matplotlib.pyplot as plt
from config import *

# Fallback to standard vector_engine if Numba version isn't created yet
try:
    from vector_engine import VectorSimulationNumba as VectorSimulation
    ENGINE_NAME = "Numba Parallel Vector Tensor Engine"
except ImportError:
    from vector_engine import VectorSimulation
    ENGINE_NAME = "Standard NumPy Vector Tensor Engine"

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

def gompertz_model(t, V0, K, a):
    """Gompertz Growth Equation: V(t) = K * exp(ln(V0/K) * exp(-a*t))"""
    return K * np.exp(np.log(V0 / K) * np.exp(-a * t))

def run_simulation_step_by_step_and_plot(days_to_run=19):
    total_steps = days_to_run * STEPS_PER_DAY

    print("=========================================================================================")
    print(f"            RUNNING {ENGINE_NAME.upper()} ({days_to_run} DAYS / {total_steps} STEPS)")
    print("=========================================================================================")
    print(f" Workspace Bounds   : {GRID_SIZE}x{GRID_SIZE}x{GRID_SIZE} ({GRID_SIZE**3:,} Spatial Voxels)")
    print(f" Initial Seed State : A={NUM_A_INIT}, B={NUM_B_INIT}, C={NUM_C_INIT}, D={NUM_D_INIT}, E={NUM_E_INIT}")
    print("=========================================================================================\n")

    sim = VectorSimulation()
    sim.seed(NUM_A_INIT, NUM_B_INIT, NUM_C_INIT, NUM_D_INIT, NUM_E_INIT)

    # Trackers for Plotting
    days_recorded = [0]
    
    a_cnt, b_cnt, c_cnt, d_n, d_a, d_e, e_cnt = sim.get_counts()
    c_vol_0, t_vol_0 = calculate_tumor_volumes(a_cnt, b_cnt, c_cnt, d_n + d_a + d_e, e_cnt)
    
    cancer_volumes = [c_vol_0]
    total_volumes = [t_vol_0]

    t0_global = time.perf_counter()
    t_day_start = t0_global

    print("-----------------------------------------------------------------------------------------")
    print("Step      | Bio Time | Agent A | Agent B | Agent C | Tcyto(N/A/E) | Treg | Cancer Vol | Step Time")
    print("-----------------------------------------------------------------------------------------")

    for step in range(1, total_steps + 1):
        t_step_0 = time.perf_counter()
        sim.step()
        dt_step = time.perf_counter() - t_step_0

        # Query counts at EVERY step
        a_cnt, b_cnt, c_cnt, d_n, d_a, d_e, e_cnt = sim.get_counts()
        d_tot = d_n + d_a + d_e
        c_vol, total_vol = calculate_tumor_volumes(a_cnt, b_cnt, c_cnt, d_tot, e_cnt)

        bio_hours = step * (TIME_STEP_MIN / 60.0)

        # Print Real-Time Step Output
        print(f"Step {step:4d}/{total_steps} | {bio_hours:5.1f}h   | "
              f"A:{a_cnt:5d} | B:{b_cnt:5d} | C:{c_cnt:4d} | "
              f"D:({d_n:2d}/{d_a:2d}/{d_e:2d})  | E:{e_cnt:3d} | "
              f"{c_vol:.4f} mm³ | {dt_step*1000:6.1f} ms")

        # Record Daily Values for Graph Plotting
        if step % STEPS_PER_DAY == 0:
            day_num = step // STEPS_PER_DAY
            dt_day = time.perf_counter() - t_day_start
            
            days_recorded.append(day_num)
            cancer_volumes.append(c_vol)
            total_volumes.append(total_vol)

            print("-----------------------------------------------------------------------------------------")
            print(f" >>> DAY {day_num:2d} CHECKPOINT | Total Cancer Pop: {a_cnt+b_cnt:,} | "
                  f"Cancer Vol: {c_vol:.4f} mm³ | Day Runtime: {dt_day:.2f}s <<<")
            print("-----------------------------------------------------------------------------------------\n")
            t_day_start = time.perf_counter()

    t_total = time.perf_counter() - t0_global
    print(f"\nSimulation completed in {t_total:.2f} seconds ({t_total/60.0:.2f} minutes).\n")

    # -------------------------------------------------------------------------
    # ANALYTICAL GOMPERTZ REFERENCE CURVES SETUP
    # -------------------------------------------------------------------------
    t_continuous = np.linspace(0, days_to_run, 200)
    
    V0 = cancer_volumes[0] if cancer_volumes[0] > 0 else 0.0335
    K = 12.7  # Carrying Capacity in mm³
    a_25 = 0.23
    a_75 = 0.28

    gomp_25 = gompertz_model(t_continuous, V0, K, a_25)
    gomp_75 = gompertz_model(t_continuous, V0, K, a_75)

    # -------------------------------------------------------------------------
    # MATPLOTLIB GRAPH GENERATION
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 6), dpi=300)
    
    plt.fill_between(t_continuous, gomp_25, gomp_75, color='#e0e0e0', alpha=0.5, label='Target Interquartile Range')
    plt.plot(t_continuous, gomp_25, color='navy', linestyle='--', linewidth=1.5, label='Gompertz 25th Percentile Target')
    plt.plot(t_continuous, gomp_75, color='crimson', linestyle='--', linewidth=1.5, label='Gompertz 75th Percentile Target')

    plt.plot(days_recorded, cancer_volumes, color='forestgreen', marker='o', markersize=5, linewidth=1.8, label='ABM Cancer-only Volume')
    plt.plot(days_recorded, total_volumes, color='goldenrod', marker='d', markersize=4, linestyle='-', linewidth=1.2, label='ABM Total Tumor Volume')

    plt.title('Validation: Simulations_Curve vs. Gompertz_Curve', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Simulation Days', fontsize=11, fontweight='bold')
    plt.ylabel('Tumor Volume (mm³)', fontsize=11, fontweight='bold')
    
    plt.xlim(-0.5, days_to_run + 0.5)
    plt.ylim(-0.2, K + 1.5)
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', fontsize=8.5, frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()

    output_filename = 'vectorized_vs_gompertz_validation.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Graph successfully generated and saved as '{output_filename}'!")
    plt.show()

if __name__ == "__main__":
    run_simulation_step_by_step_and_plot(days_to_run=19)