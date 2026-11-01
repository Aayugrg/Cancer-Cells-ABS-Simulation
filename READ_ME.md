====================================================================================================
           3D AGENT-BASED CANCER SIMULATION SUITE (VECTORIZED & ACCELERATED TENSOR ENGINE)
====================================================================================================

OVERVIEW
--------
This research software suite provides a high-performance 3D Agent-Based Model (ABM) simulating tumor
microenvironment dynamics across 5 biological agent types (7 total functional phenotypes) and a 3D
CXCL12 chemokine diffusion field.

By replacing traditional CPython object-oriented loops with a Unified 4D Tensor State Engine 
(S in R^(150 x 150 x 150 x 12)) parallelized via Numba JIT compilation (@njit(parallel=True)), 
the simulation achieves high compute throughput on standard multi-core laptops. Visual progress, 
HUD telemetry, and automated 2D/3D snapshots are managed via VisPy OpenGL rendering and Matplotlib.


====================================================================================================
1. REPOSITORY FILE ARCHITECTURE
====================================================================================================

Cancer_Simulation_Suite/
│
├── config.py                 # Central configuration: parameters, thresholds, and tensor maps
├── vector_engine.py          # Numba JIT multi-threaded 4D Tensor Engine (Bug-free execution)
├── naive_engine.py           # Reference Object-Oriented (OOP) baseline engine (v13 logic)
├── renderer.py               # VisPy 3D OpenGL visualization window, legend & HUD overlays
│
├── main_live.py              # Runner 1: Real-time live parallel simulation + snapshots
├── main_offline.py           # Runner 2: Pre-compute trajectory, cache .npz & 60 FPS replay
├── benchmark.py              # Performance & output parity validator (OOP vs. Tensor Engine)
└── plot_gompertz_comparison.py# Analytical validation plot generator vs. Gompertz growth limits


====================================================================================================
2. PREREQUISITES & DEPENDENCIES
====================================================================================================

This suite runs entirely on CPU multi-core architectures (Intel / AMD) without requiring a dedicated
NVIDIA GPU.

Python Version: Python 3.9 or higher

Required Packages:
To install all required packages, run the following command in your terminal:

    pip install numpy scipy numba vispy matplotlib PyQt6

Note: VisPy requires an interactive GUI backend. PyQt6 is recommended.


====================================================================================================
3. HOW TO OPERATE & RUN THE SIMULATION
====================================================================================================

[A] REAL-TIME LIVE PARALLEL SIMULATION (main_live.py)
-----------------------------------------------------
Steps the Numba-accelerated biological engine in real-time while streaming coordinate buffers directly
to the 3D VisPy OpenGL viewport.

Command:
    python main_live.py

Features:
  - Live HUD displaying Day, Cancer Vol (mm³), and Total Vol (mm³).
  - Phenotype color key legend displayed in top-left window corner.
  - Interactive 3D camera (rotate, pan, zoom).
  - Automated exports of 3D perspective screenshots and 2D cross-sectional Z-slices every K hours 
    saved to: ./simulation_snapshots_live/


[B] HIGH-SPEED PRE-COMPUTATION & REPLAY (main_offline.py)
--------------------------------------------------------
Computes and caches the full 40-day trajectory to a compressed file (tumor_trajectory_40days.npz) 
and launches an interactive 60 FPS replay window.

Command:
    python main_offline.py

Features:
  - Instant, lag-free 60 FPS playback during dense tumor phases (>150,000 cells).
  - Pre-computes and exports 2D midline spatial slices and 3D VisPy frame renders every K hours
    saved to: ./simulation_snapshots_offline/


[C] PERFORMANCE BENCHMARK & PARITY VALIDATION (benchmark.py)
-------------------------------------------------------------
Executes both the Naive OOP Reference Engine and the Vectorized Engine step-by-step to measure execution
speedups and verify mathematical parity across all diagnostic counters.

Command:
    python benchmark.py


[D] GOMPERTZ CURVE VALIDATION PLOTTER (plot_gompertz_comparison.py)
-------------------------------------------------------------------
Runs the vectorized simulation, extracts daily tumor volume metrics, computes analytical 25th & 75th
percentile Gompertz bounds for breast cancer, and outputs a publication-ready validation figure.

Command:
    python plot_gompertz_comparison.py
Output:
    vectorized_vs_gompertz_validation.png


====================================================================================================
4. PARAMETER CONFIGURATION GUIDE
====================================================================================================

All global simulation parameters can be customized across the source files:

1. WORKSPACE & TIME HORIZON (config.py)
---------------------------------------
  - GRID_SIZE = 150             : Spatial grid dimensions (150x150x150 = 3,375,000 voxels, 20µm/voxel)
  - MOLECULAR_GRID_SIZE = 300   : High-res CXCL12 diffusion grid (300x300x300, 10µm resolution)
  - TOTAL_DAYS = 40             : Simulation duration horizon
  - TIME_STEP_MIN = 30          : Time step duration (30 minutes; 48 steps = 1 biological day)

2. AUTOMATED SNAPSHOT & 2D SLICE SETTINGS (main_live.py & main_offline.py)
--------------------------------------------------------------------------
  - SNAPSHOT_INTERVAL_HOURS = 24.0 : Export snapshots every K hours (e.g., 12.0, 24.0, 48.0)
  - SLICE_Z_TARGET = 75            : Target Z-plane for 2D slice (0 to 149; 75 = midline center)
  - SAVE_SNAPSHOTS = True          : Set to True to enable disk exports (False to disable)

3. INITIAL SEED POPULATIONS (config.py)
---------------------------------------
  - NUM_A_INIT = 800            : Agent A (CXCR4-ve Cancer Cells)
  - NUM_B_INIT = 2400           : Agent B (CXCR4+ve Cancer Cells)
  - NUM_C_INIT = 120            : Agent C (Stromal CAFs)
  - NUM_D_INIT = 8              : Agent D (Cytotoxic T-Cells)
  - NUM_E_INIT = 3              : Agent E (Regulatory T-Cells / Treg)

4. BIOLOGICAL MECHANICS & THRESHOLDS (config.py)
-----------------------------------------------
  - P_A_TO_B_PER_DAY = 0.20     : Phenotype switch probability (A -> B: 20%/day)
  - P_B_TO_A_PER_DAY = 0.10     : Phenotype switch probability (B -> A: 10%/day)
  - KD_CXCL12 = 19.0            : Chemokine threshold (nM) for Agent B division acceleration
  - CONTACT_THRESHOLD_C = 24    : 12 hours of cancer contact triggers 4x CXCL12 stromal secretion
  - CONTACT_ACT_THRESHOLD_D = 8 : 4 hours of cancer contact triggers Cytotoxic T-cell TCR activation


====================================================================================================
5. FEATURE MATRIX CHANNEL MAPPING (N = 12)
====================================================================================================

Every spatial voxel (x, y, z) maps to a 12-channel 32-bit floating-point feature vector in tensor S:

  Channel Index | Constant Name      | Encoding / Value Description
  ------------- | ------------------ | -------------------------------------------------------------
       0        | OCCUPIED           | 0.0 = Empty slot, 1.0 = Occupied slot
       1        | CELL_TYPE          | 1.0=A, 2.0=B, 3.0=C, 4.0=D, 5.0=E
       2        | SUB_STATE          | For D: 0.0=Naive, 1.0=Activated, 2.0=Exhausted
       3        | REMAINING_LIFESPAN | Gaussian lifespan step countdown ~ N(20000, 96)
       4        | STEP_COUNTER       | Age / Action timer tracking division/migration intervals
       5        | DIVISIONS          | Clonal division counter
       6        | WAITING_DIV        | 1.0 = Blocked by local crowding, awaiting empty neighbor
       7        | CONTACT_STEPS      | Continuous cell-cell contact duration timer
       8        | SECRETION_STATE    | Stromal secretion state (1.0 = Baseline, 4.0 = Activated)
       9        | FIRST_DIV_DELAY    | Delay timer prior to first post-activation division (48..96)
      10        | LAST_DIV_STEP      | Step timestamp of previous division
      11        | P_KILL_EXHAUSTED   | Individual kill probability drawn for exhausted T-cells


====================================================================================================
6. COLOR PALETTE & PHENOTYPE MAP
====================================================================================================

  Color Code   | Entity / Phenotype         | Biological Function
  ------------ | -------------------------- | --------------------------------------------------
  Red          | Agent A (CXCR4-ve)         | Primary un-stimulated cancer cell
  Orange       | Agent B (CXCR4+ve)         | Chemokine-sensitive cancer cell (Accelerates at >19nM)
  Green        | Agent C (Stromal CAF)      | Secretes CXCL12 chemokine scaffold
  Light Blue   | Agent D (Tcyto Naive)      | Surveying cytotoxic T-cell
  Cyan         | Agent D (Tcyto Activated)  | Actively killing cytotoxic T-cell
  Purple       | Agent D (Tcyto Exhausted)  | Functional immune exhaustion
  Yellow       | Agent E (Treg)             | Regulatory T-cell (Drives exhaustion / chemotaxis)


====================================================================================================
7. INTERACTIVE 3D VIEWPORT CONTROLS (VisPy Window)
====================================================================================================

  Action               | Mouse / Keyboard Input
  -------------------- | -------------------------------------------------------------------
  Rotate 3D Mass       | Click and drag Left Mouse Button
  Pan Camera           | Hold 'Shift' + Click and drag Left Mouse Button (or Middle Button)
  Zoom In / Out        | Scroll Mouse Wheel
  Reset Camera View    | Press 'Spacebar'
====================================================================================================