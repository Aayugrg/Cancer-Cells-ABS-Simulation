import math

# Workspace & Horizon Settings
GRID_SIZE = 150                      # 150x150x150 = 3,375,000 Voxels
MOLECULAR_GRID_SIZE = GRID_SIZE * 2  # 300x300x300 (10µm resolution)

TIME_STEP_MIN = 30
STEPS_PER_DAY = 48
TOTAL_DAYS = 40                      # Full 40-Day Study Horizon
TOTAL_STEPS = TOTAL_DAYS * STEPS_PER_DAY

# Initial Seed Populations
NUM_A_INIT = 800
NUM_B_INIT = 2400
NUM_C_INIT = 120
NUM_D_INIT = 8
NUM_E_INIT = 3

# Chemokine Parameters
D_CXCL12 = 120.0        # µm^2/sec
HALF_LIFE_MIN = 26.5    # minutes
KD_CXCL12 = 19.0        # nM threshold
TIME_STEP_SEC = TIME_STEP_MIN * 60
DECAY_FACTOR = math.exp(-math.log(2) * TIME_STEP_MIN / HALF_LIFE_MIN)

# Cancer Agents (A: CXCR4-ve, B: CXCR4+ve)
PROLIF_STEPS_A = 52
MIGRATE_STEPS_A = 2
BASE_PROLIF_STEPS_B = 52
ACT_PROLIF_STEPS_B = 36
BASE_MIGRATE_STEPS_B = 2
ACT_MIGRATE_STEPS_B = 1

MU_LIFESPAN_STEPS = 20000
SIGMA_LIFESPAN_STEPS = 96
P_PROLIF_BASE = 0.32
GAMMA = 1.0

P_A_TO_B_PER_DAY = 0.20
P_A_TO_B_PER_STEP = 1.0 - (1.0 - P_A_TO_B_PER_DAY) ** (1.0 / STEPS_PER_DAY)
P_B_TO_A_PER_DAY = 0.10
P_B_TO_A_PER_STEP = 1.0 - (1.0 - P_B_TO_A_PER_DAY) ** (1.0 / STEPS_PER_DAY)

# Stromal Agent C
PROLIF_STEPS_C = 240
MIGRATE_STEPS_C = 2

# T-Cell Agent D & E
MIGRATE_STEPS_D = 5
MIGRATE_STEPS_E = 4
PROLIF_STEPS_E = 30

# Feature Matrix Index Map (N = 12)
NUM_FEATURES = 12

OCCUPIED           = 0   # 0.0 = Empty, 1.0 = Occupied
CELL_TYPE          = 1   # 1=A, 2=B, 3=C, 4=D, 5=E
SUB_STATE          = 2   # For D: 0=Naive, 1=Activated, 2=Exhausted
REMAINING_LIFESPAN = 3
STEP_COUNTER       = 4
DIVISIONS          = 5
WAITING_DIV        = 6
CONTACT_STEPS      = 7
SECRETION_STATE    = 8
FIRST_DIV_DELAY    = 9
LAST_DIV_STEP      = 10
P_KILL_EXHAUSTED   = 11

# VisPy Color Palette (RGBA Normalized)
COLOR_MAP = {
    1: [1.0, 0.2, 0.2, 0.8],   # Red     -> Agent A (CXCR4-ve)
    2: [1.0, 0.5, 0.0, 0.9],   # Orange  -> Agent B (CXCR4+ve)
    3: [0.2, 0.8, 0.2, 0.5],   # Green   -> Agent C (Stromal)
    40: [0.2, 0.6, 1.0, 0.9],  # Light Blue -> Agent D (Naive)
    41: [0.0, 1.0, 1.0, 1.0],  # Cyan    -> Agent D (Activated)
    42: [0.6, 0.0, 1.0, 0.8],  # Purple  -> Agent D (Exhausted)
    5: [1.0, 1.0, 0.0, 0.9]    # Yellow  -> Agent E (Treg)
}