import time
import math
from config import *
from naive_engine import NaiveSimulation
from vector_engine import VectorSimulation

def calculate_volumes(count_a, count_b, count_c, count_d, count_e):
    """Calculates tumor volumes in mm³ according to physical voxel radii."""
    pi_val = math.pi
    # Cancer cells (A + B): r = 10 µm
    vol_cancer_voxels = (count_a + count_b) * ((4/3) * pi_val * (10**3))
    # Stromal & Immune cells (C + D + E): r = 5 µm
    vol_other_voxels = (count_c + count_d + count_e) * ((4/3) * pi_val * (5**3))
    
    cancer_vol_mm3 = vol_cancer_voxels / (10**9)
    total_vol_mm3 = (vol_cancer_voxels + vol_other_voxels) / (10**9)
    return cancer_vol_mm3, total_vol_mm3


def run_realtime_benchmark():
    print("=========================================================================================")
    print("          STEP-BY-STEP SIMULATION BENCHMARK: OOP vs UNIFIED VECTOR TENSOR ENGINE         ")
    print("=========================================================================================")
    print(f" Timeline Horizon   : {BENCHMARK_DAYS} Days ({TOTAL_STEPS} Steps @ 30 min/step)")
    print(f" Workspace Bounds   : {GRID_SIZE}x{GRID_SIZE}x{GRID_SIZE} ({GRID_SIZE**3:,} Spatial Voxels)")
    print(f" Initial Seed State : A={NUM_A_INIT}, B={NUM_B_INIT}, C={NUM_C_INIT}, D={NUM_D_INIT}, E={NUM_E_INIT}")
    print("=========================================================================================\n")

    
    # -----------------------------------------------------------------------------------------
    # 2. UNIFIED VECTOR TENSOR ENGINE EXECUTION
    # -----------------------------------------------------------------------------------------
    print("-----------------------------------------------------------------------------------------")
    print("[2/2] RUNNING UNIFIED VECTOR TENSOR ENGINE (Step-by-Step Progress)")
    print("-----------------------------------------------------------------------------------------")

    vec_sim = VectorSimulation()
    vec_sim.seed(NUM_A_INIT, NUM_B_INIT, NUM_C_INIT, NUM_D_INIT, NUM_E_INIT)

    t0_vec = time.perf_counter()
    day_start_time = t0_vec

    for step in range(1, TOTAL_STEPS + 1):
        step_t0 = time.perf_counter()
        vec_sim.step()
        step_dt = time.perf_counter() - step_t0

        # Print detailed step progress
        ve_a, ve_b, ve_c, ve_dn, ve_da, ve_de, ve_e = vec_sim.get_counts()
        ve_d_total = ve_dn + ve_da + ve_de
        c_vol, t_vol = calculate_volumes(ve_a, ve_b, ve_c, ve_d_total, ve_e)

        current_hours = step * 0.5
        print(f"  [VEC] Step {step:3d}/{TOTAL_STEPS} | Time: {current_hours:5.1f}h | "
              f"A:{ve_a:5d} B:{ve_b:5d} C:{ve_c:4d} D:{ve_d_total:3d} E:{ve_e:3d} | "
              f"Cancer Vol: {c_vol:.4f} mm³ | Step Time: {step_dt*1000:6.2f} ms")

        # End-of-Day Checkpoint
        if step % STEPS_PER_DAY == 0:
            day_num = step // STEPS_PER_DAY
            day_dt = time.perf_counter() - day_start_time
            print(f"\n  >>> DAY {day_num} COMPLETED in {day_dt:.3f} sec | "
                  f"Total Cancer: {ve_a + ve_b:,} | Total Cells: {ve_a + ve_b + ve_c + ve_d_total + ve_e:,} <<<\n")
            day_start_time = time.perf_counter()

    vec_total_time = time.perf_counter() - t0_vec
    print(f"  └─ Vector Engine Completed in: {vec_total_time:.3f} seconds\n")

    # -----------------------------------------------------------------------------------------
    # 1. NAIVE OOP ENGINE EXECUTION
    # -----------------------------------------------------------------------------------------
    print("-----------------------------------------------------------------------------------------")
    print("[1/2] RUNNING NAIVE OOP ENGINE (Step-by-Step Progress)")
    print("-----------------------------------------------------------------------------------------")
    
    naive_sim = NaiveSimulation()
    naive_sim.seed(NUM_A_INIT, NUM_B_INIT, NUM_C_INIT, NUM_D_INIT, NUM_E_INIT)

    t0_naive = time.perf_counter()
    day_start_time = t0_naive

    for step in range(1, TOTAL_STEPS + 1):
        step_t0 = time.perf_counter()
        naive_sim.step()
        step_dt = time.perf_counter() - step_t0

        # Print detailed step progress
        na_a, na_b, na_c, na_dn, na_da, na_de, na_e = naive_sim.get_counts()
        na_d_total = na_dn + na_da + na_de
        c_vol, t_vol = calculate_volumes(na_a, na_b, na_c, na_d_total, na_e)

        current_hours = step * 0.5
        print(f"  [OOP] Step {step:3d}/{TOTAL_STEPS} | Time: {current_hours:5.1f}h | "
              f"A:{na_a:5d} B:{na_b:5d} C:{na_c:4d} D:{na_d_total:3d} E:{na_e:3d} | "
              f"Cancer Vol: {c_vol:.4f} mm³ | Step Time: {step_dt*1000:6.2f} ms")

        # End-of-Day Checkpoint
        if step % STEPS_PER_DAY == 0:
            day_num = step // STEPS_PER_DAY
            day_dt = time.perf_counter() - day_start_time
            print(f"\n  >>> DAY {day_num} COMPLETED in {day_dt:.3f} sec | "
                  f"Total Cancer: {na_a + na_b:,} | Total Cells: {na_a + na_b + na_c + na_d_total + na_e:,} <<<\n")
            day_start_time = time.perf_counter()

    naive_total_time = time.perf_counter() - t0_naive
    print(f"  └─ OOP Engine Completed in: {naive_total_time:.3f} seconds\n")


    # -----------------------------------------------------------------------------------------
    # 3. PERFORMANCE SUMMARY & PARITY REPORT
    # -----------------------------------------------------------------------------------------
    speedup = naive_total_time / vec_total_time if vec_total_time > 0 else 0.0

    print("=========================================================================================")
    print("                               PERFORMANCE METRICS SUMMARY                               ")
    print("=========================================================================================")
    print(f" Naive OOP Runtime        : {naive_total_time:.3f} seconds")
    print(f" Unified Vector Runtime   : {vec_total_time:.3f} seconds")
    print(f" Performance Acceleration : {speedup:.2f}x Speedup")
    print("=========================================================================================")

    print("\n=========================================================================================")
    print("                                OUTPUT PARITY VERIFICATION                               ")
    print("=========================================================================================")
    print(f" Entity / Diagnostic Metric  | Naive OOP   | Unified Vector")
    print(f" ----------------------------|-------------|------------------")
    print(f" Final Agent A (CXCR4-ve)    | {na_a:<11} | {ve_a}")
    print(f" Final Agent B (CXCR4+ve)    | {na_b:<11} | {ve_b}")
    print(f" Final Agent C (Stromal)     | {na_c:<11} | {ve_c}")
    print(f" Final Agent D (Tcyto-Naive) | {na_dn:<11} | {ve_dn}")
    print(f" Final Agent D (Tcyto-Act)   | {na_da:<11} | {ve_da}")
    print(f" Final Agent D (Tcyto-Exh)   | {na_de:<11} | {ve_de}")
    print(f" Final Agent E (Treg)        | {na_e:<11} | {ve_e}")
    print(f" Prolif Succeeded            | {naive_sim.diagnostics['Prolif_Succeeded']:<11} | {vec_sim.diagnostics['Prolif_Succeeded']}")
    print(f" Prolif Gated (Probability)  | {naive_sim.diagnostics['Prolif_ProbGateFailed']:<11} | {vec_sim.diagnostics['Prolif_ProbGateFailed']}")
    print(f" Natural Lifespan Deaths     | {naive_sim.diagnostics['Death_NaturalLifespan']:<11} | {vec_sim.diagnostics['Death_NaturalLifespan']}")
    print(f" Phenotype Switch Events     | {naive_sim.diagnostics['Death_PhenotypeSwitch']:<11} | {vec_sim.diagnostics['Death_PhenotypeSwitch']}")
    print("=========================================================================================\n")


if __name__ == "__main__":
    run_realtime_benchmark()