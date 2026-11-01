import numpy as np
import random
import math
from numba import njit, prange
from config import *

# -----------------------------------------------------------------------------
# NUMBA JIT PARALLEL KERNELS 
# -----------------------------------------------------------------------------

@njit(parallel=True)
def numba_update_lifespan_and_switch(state, p_a_to_b, p_b_to_a):
    Z, Y, X, _ = state.shape
    num_dead = 0
    num_switches = 0

    for z in prange(Z):
        for y in range(Y):
            for x in range(X):
                if state[z, y, x, OCCUPIED] == 1.0:
                    c_type = state[z, y, x, CELL_TYPE]

                    # 1. Lifespan Decay
                    if c_type == 1.0 or c_type == 2.0:
                        state[z, y, x, REMAINING_LIFESPAN] -= 1.0
                        if state[z, y, x, REMAINING_LIFESPAN] <= 0.0:
                            for f in range(NUM_FEATURES):
                                state[z, y, x, f] = 0.0
                            num_dead += 1
                            continue

                    # 2. Phenotype Switching
                    if c_type == 1.0:
                        if random.random() < p_a_to_b:
                            state[z, y, x, CELL_TYPE] = 2.0
                            num_switches += 1
                    elif c_type == 2.0:
                        if random.random() < p_b_to_a:
                            state[z, y, x, CELL_TYPE] = 1.0
                            num_switches += 1

                    # 3. Increment Age Step Counter
                    state[z, y, x, STEP_COUNTER] += 1.0

    return num_dead, num_switches


@njit(parallel=True)
def numba_convolve_3d_occupancy(state):
    Z, Y, X, _ = state.shape
    neighbor_counts = np.zeros((Z, Y, X), dtype=np.uint8)

    for z in prange(1, Z - 1):
        for y in range(1, Y - 1):
            for x in range(1, X - 1):
                count = 0
                for dz in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dz == 0 and dy == 0 and dx == 0:
                                continue
                            if state[z + dz, y + dy, x + dx, OCCUPIED] == 1.0:
                                count += 1
                neighbor_counts[z, y, x] = count

    return neighbor_counts


@njit
def numba_execute_proliferation_and_migration(state, was_occupied, neighbor_counts, p_prolif_base, gamma,
                                             prolif_a, prolif_b, prolif_c, prolif_e,
                                             mig_a, mig_b, mig_c, mig_d, mig_e,
                                             mu_life, sigma_life):
    Z, Y, X, _ = state.shape
    succ_divs = 0
    prob_failed = 0
    space_blocked = 0

    offsets = [
        (-1,-1,-1), (-1,-1, 0), (-1,-1, 1), (-1, 0,-1), (-1, 0, 0), (-1, 0, 1), (-1, 1,-1), (-1, 1, 0), (-1, 1, 1),
        ( 0,-1,-1), ( 0,-1, 0), ( 0,-1, 1), ( 0, 0,-1),           ( 0, 0, 1), ( 0, 1,-1), ( 0, 1, 0), ( 0, 1, 1),
        ( 1,-1,-1), ( 1,-1, 0), ( 1,-1, 1), ( 1, 0,-1), ( 1, 0, 0), ( 1, 0, 1), ( 1, 1,-1), ( 1, 1, 0), ( 1, 1, 1)
    ]

    for z in range(1, Z - 1):
        for y in range(1, Y - 1):
            for x in range(1, X - 1):
                # CRITICAL FIX: Only process cells that were occupied AT THE START of this step
                if was_occupied[z, y, x] == 1.0 and state[z, y, x, OCCUPIED] == 1.0:
                    c_type = state[z, y, x, CELL_TYPE]
                    steps = state[z, y, x, STEP_COUNTER]

                    div_interval = 0
                    if c_type == 1.0: div_interval = prolif_a
                    elif c_type == 2.0: div_interval = prolif_b
                    elif c_type == 3.0: div_interval = prolif_c
                    elif c_type == 5.0: div_interval = prolif_e

                    # --- PROLIFERATION ---
                    if div_interval > 0 and (steps % div_interval == 0 or state[z, y, x, WAITING_DIV] == 1.0):
                        n_occ = neighbor_counts[z, y, x]
                        empty_count = 26 - n_occ

                        if empty_count == 0:
                            state[z, y, x, WAITING_DIV] = 1.0
                            space_blocked += 1
                        else:
                            local_occ = n_occ / 26.0
                            p_succ = p_prolif_base * ((1.0 - local_occ) ** gamma)

                            if random.random() <= p_succ:
                                empties_z = []
                                empties_y = []
                                empties_x = []
                                for dz, dy, dx in offsets:
                                    nz, ny, nx = z + dz, y + dy, x + dx
                                    if state[nz, ny, nx, OCCUPIED] == 0.0:
                                        empties_z.append(nz)
                                        empties_y.append(ny)
                                        empties_x.append(nx)

                                if len(empties_z) > 0:
                                    idx = random.randint(0, len(empties_z) - 1)
                                    nz, ny, nx = empties_z[idx], empties_y[idx], empties_x[idx]

                                    state[nz, ny, nx, OCCUPIED] = 1.0
                                    state[nz, ny, nx, CELL_TYPE] = c_type
                                    state[nz, ny, nx, STEP_COUNTER] = 0.0

                                    if c_type == 1.0 or c_type == 2.0:
                                        u1, u2 = random.random(), random.random()
                                        g_z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
                                        state[nz, ny, nx, REMAINING_LIFESPAN] = max(1.0, mu_life + sigma_life * g_z)

                                    state[z, y, x, WAITING_DIV] = 0.0
                                    succ_divs += 1
                            else:
                                state[z, y, x, WAITING_DIV] = 1.0
                                prob_failed += 1

                    # --- MIGRATION ---
                    mig_interval = 0
                    if c_type == 1.0: mig_interval = mig_a
                    elif c_type == 2.0: mig_interval = mig_b
                    elif c_type == 3.0: mig_interval = mig_c
                    elif c_type == 4.0: mig_interval = mig_d
                    elif c_type == 5.0: mig_interval = mig_e

                    if mig_interval > 0 and steps % mig_interval == 0 and state[z, y, x, WAITING_DIV] == 0.0:
                        empties_z = []
                        empties_y = []
                        empties_x = []
                        for dz, dy, dx in offsets:
                            nz, ny, nx = z + dz, y + dy, x + dx
                            if state[nz, ny, nx, OCCUPIED] == 0.0:
                                empties_z.append(nz)
                                empties_y.append(ny)
                                empties_x.append(nx)

                        if len(empties_z) > 0:
                            idx = random.randint(0, len(empties_z) - 1)
                            nz, ny, nx = empties_z[idx], empties_y[idx], empties_x[idx]

                            if nz == 0 or ny == 0 or nx == 0 or nz == Z - 1 or ny == Y - 1 or nx == X - 1:
                                for f in range(NUM_FEATURES):
                                    state[z, y, x, f] = 0.0
                            else:
                                for f in range(NUM_FEATURES):
                                    state[nz, ny, nx, f] = state[z, y, x, f]
                                    state[z, y, x, f] = 0.0

    return succ_divs, prob_failed, space_blocked


class VectorSimulationNumba:
    def __init__(self):
        self.State = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE, NUM_FEATURES), dtype=np.float32)
        self.cxcl12_grid = np.zeros((MOLECULAR_GRID_SIZE, MOLECULAR_GRID_SIZE, MOLECULAR_GRID_SIZE), dtype=np.float32)

        self.diagnostics = {
            "Prolif_Succeeded": 0, "Prolif_NoEmptyNeighbors": 0,
            "Prolif_ProbGateFailed": 0, "Death_NaturalLifespan": 0,
            "Death_PhenotypeSwitch": 0
        }

    def seed(self, num_a, num_b, num_c, num_d, num_e):
        total_seeded = num_a + num_b + num_c + num_d + num_e
        indices = np.random.choice(GRID_SIZE**3, size=total_seeded, replace=False)
        coords = np.unravel_index(indices, (GRID_SIZE, GRID_SIZE, GRID_SIZE))

        i1 = num_a
        i2 = i1 + num_b
        i3 = i2 + num_c
        i4 = i3 + num_d

        c_a = (coords[0][:i1], coords[1][:i1], coords[2][:i1])
        c_b = (coords[0][i1:i2], coords[1][i1:i2], coords[2][i1:i2])
        c_c = (coords[0][i2:i3], coords[1][i2:i3], coords[2][i2:i3])
        c_d = (coords[0][i3:i4], coords[1][i3:i4], coords[2][i3:i4])
        c_e = (coords[0][i4:], coords[1][i4:], coords[2][i4:])

        self.State[c_a + (OCCUPIED,)] = 1.0; self.State[c_a + (CELL_TYPE,)] = 1.0
        self.State[c_b + (OCCUPIED,)] = 1.0; self.State[c_b + (CELL_TYPE,)] = 2.0
        self.State[c_c + (OCCUPIED,)] = 1.0; self.State[c_c + (CELL_TYPE,)] = 3.0; self.State[c_c + (SECRETION_STATE,)] = 1.0
        self.State[c_d + (OCCUPIED,)] = 1.0; self.State[c_d + (CELL_TYPE,)] = 4.0; self.State[c_d + (SUB_STATE,)] = 0.0
        self.State[c_e + (OCCUPIED,)] = 1.0; self.State[c_e + (CELL_TYPE,)] = 5.0

        cancer_coords = (coords[0][:i2], coords[1][:i2], coords[2][:i2])
        lifespans = np.random.normal(MU_LIFESPAN_STEPS, SIGMA_LIFESPAN_STEPS, num_a + num_b).astype(np.float32)
        lifespans = np.maximum(lifespans, 1.0)
        self.State[cancer_coords + (REMAINING_LIFESPAN,)] = lifespans

    def step(self):
        # 1. Lifespan Decay & Phenotype Switch
        n_dead, n_switch = numba_update_lifespan_and_switch(self.State, P_A_TO_B_PER_STEP, P_B_TO_A_PER_STEP)
        self.diagnostics["Death_NaturalLifespan"] += n_dead
        self.diagnostics["Death_PhenotypeSwitch"] += n_switch

        # 2. Snapshot occupied grid to prevent cascading divisions in the same step
        was_occupied = self.State[:, :, :, OCCUPIED].copy()

        # 3. 3D Neighbor Stencil Convolutions
        neighbor_counts = numba_convolve_3d_occupancy(self.State)

        # 4. Proliferation & Migration
        s_divs, p_fail, s_block = numba_execute_proliferation_and_migration(
            self.State, was_occupied, neighbor_counts, P_PROLIF_BASE, GAMMA,
            PROLIF_STEPS_A, BASE_PROLIF_STEPS_B, PROLIF_STEPS_C, PROLIF_STEPS_E,
            MIGRATE_STEPS_A, BASE_MIGRATE_STEPS_B, MIGRATE_STEPS_C, MIGRATE_STEPS_D, MIGRATE_STEPS_E,
            MU_LIFESPAN_STEPS, SIGMA_LIFESPAN_STEPS
        )

        self.diagnostics["Prolif_Succeeded"] += s_divs
        self.diagnostics["Prolif_ProbGateFailed"] += p_fail
        self.diagnostics["Prolif_NoEmptyNeighbors"] += s_block

        # 5. Chemokine Field Decay
        self.cxcl12_grid *= DECAY_FACTOR

    def get_counts(self):
        active = self.State[:, :, :, OCCUPIED] == 1.0
        a_count = np.sum(active & (self.State[:, :, :, CELL_TYPE] == 1.0))
        b_count = np.sum(active & (self.State[:, :, :, CELL_TYPE] == 2.0))
        c_count = np.sum(active & (self.State[:, :, :, CELL_TYPE] == 3.0))

        tcyto_mask = active & (self.State[:, :, :, CELL_TYPE] == 4.0)
        d_naive = np.sum(tcyto_mask & (self.State[:, :, :, SUB_STATE] == 0.0))
        d_act   = np.sum(tcyto_mask & (self.State[:, :, :, SUB_STATE] == 1.0))
        d_exh   = np.sum(tcyto_mask & (self.State[:, :, :, SUB_STATE] == 2.0))

        e_count = np.sum(active & (self.State[:, :, :, CELL_TYPE] == 5.0))
        return int(a_count), int(b_count), int(c_count), int(d_naive), int(d_act), int(d_exh), int(e_count)

    def extract_render_buffers(self):
        """Extracts (N, 3) point positions and (N, 4) color buffers directly from the 4D state tensor."""
        active_coords = np.argwhere(self.State[:, :, :, OCCUPIED] == 1.0)
        if len(active_coords) == 0:
            return np.empty((0, 3)), np.empty((0, 4))

        z, y, x = active_coords[:, 0], active_coords[:, 1], active_coords[:, 2]
        types = self.State[z, y, x, CELL_TYPE].astype(int)
        sub_states = self.State[z, y, x, SUB_STATE].astype(int)

        positions = np.column_stack((x, y, z)).astype(np.float32)
        colors = np.zeros((len(positions), 4), dtype=np.float32)

        for i in range(len(positions)):
            t = types[i]
            if t == 4:
                key = 40 + sub_states[i]
            else:
                key = t
            colors[i] = COLOR_MAP.get(key, [1.0, 1.0, 1.0, 0.5])

        return positions, colors





import numpy as np
from scipy.ndimage import convolve
from config import *

# 3x3x3 Stencil Kernel for 26-neighborhood
KERNEL_26 = np.ones((3, 3, 3), dtype=np.uint8)
KERNEL_26[1, 1, 1] = 0

class VectorSimulation:
    def __init__(self):
        self.State = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE, NUM_FEATURES), dtype=np.float32)
        self.cxcl12_grid = np.zeros((MOLECULAR_GRID_SIZE, MOLECULAR_GRID_SIZE, MOLECULAR_GRID_SIZE), dtype=np.float32)

        self.diagnostics = {
            "Prolif_Succeeded": 0, "Prolif_NoEmptyNeighbors": 0,
            "Prolif_ProbGateFailed": 0, "Death_NaturalLifespan": 0,
            "Death_PhenotypeSwitch": 0
        }

    def seed(self, num_a, num_b, num_c, num_d, num_e):
        total_seeded = num_a + num_b + num_c + num_d + num_e
        indices = np.random.choice(GRID_SIZE**3, size=total_seeded, replace=False)
        coords = np.unravel_index(indices, (GRID_SIZE, GRID_SIZE, GRID_SIZE))

        # Slicing coordinate tuples
        i1 = num_a
        i2 = i1 + num_b
        i3 = i2 + num_c
        i4 = i3 + num_d

        c_a = (coords[0][:i1], coords[1][:i1], coords[2][:i1])
        c_b = (coords[0][i1:i2], coords[1][i1:i2], coords[2][i1:i2])
        c_c = (coords[0][i2:i3], coords[1][i2:i3], coords[2][i2:i3])
        c_d = (coords[0][i3:i4], coords[1][i3:i4], coords[2][i3:i4])
        c_e = (coords[0][i4:], coords[1][i4:], coords[2][i4:])

        # Seed Agent A (Cancer CXCR4-ve)
        self.State[c_a + (OCCUPIED,)] = 1.0
        self.State[c_a + (CELL_TYPE,)] = 1.0

        # Seed Agent B (Cancer CXCR4+ve)
        self.State[c_b + (OCCUPIED,)] = 1.0
        self.State[c_b + (CELL_TYPE,)] = 2.0

        # Seed Agent C (Stromal)
        self.State[c_c + (OCCUPIED,)] = 1.0
        self.State[c_c + (CELL_TYPE,)] = 3.0
        self.State[c_c + (SECRETION_STATE,)] = 1.0

        # Seed Agent D (Tcyto)
        self.State[c_d + (OCCUPIED,)] = 1.0
        self.State[c_d + (CELL_TYPE,)] = 4.0
        self.State[c_d + (SUB_STATE,)] = 0.0  # Naive

        # Seed Agent E (Treg)
        self.State[c_e + (OCCUPIED,)] = 1.0
        self.State[c_e + (CELL_TYPE,)] = 5.0

        # Assign lifespans to A and B
        cancer_coords = (coords[0][:i2], coords[1][:i2], coords[2][:i2])
        lifespans = np.random.normal(MU_LIFESPAN_STEPS, SIGMA_LIFESPAN_STEPS, num_a + num_b).astype(np.float32)
        lifespans = np.maximum(lifespans, 1.0)
        self.State[cancer_coords + (REMAINING_LIFESPAN,)] = lifespans

    def step(self):
        active_mask = self.State[:, :, :, OCCUPIED] == 1.0

        # -----------------------------------------------------------------
        # 1. LIFESPAN DECAY (Cancer A & B)
        # -----------------------------------------------------------------
        cancer_mask = active_mask & ((self.State[:, :, :, CELL_TYPE] == 1.0) | (self.State[:, :, :, CELL_TYPE] == 2.0))
        self.State[:, :, :, REMAINING_LIFESPAN][cancer_mask] -= 1.0
        dead_lifespan = cancer_mask & (self.State[:, :, :, REMAINING_LIFESPAN] <= 0.0)

        num_dead = np.sum(dead_lifespan)
        if num_dead > 0:
            self.diagnostics["Death_NaturalLifespan"] += int(num_dead)
            self.State[dead_lifespan, :] = 0.0
            active_mask = self.State[:, :, :, OCCUPIED] == 1.0

        # -----------------------------------------------------------------
        # 2. PHENOTYPE SWITCHING (A <-> B)
        # -----------------------------------------------------------------
        mask_a = active_mask & (self.State[:, :, :, CELL_TYPE] == 1.0)
        mask_b = active_mask & (self.State[:, :, :, CELL_TYPE] == 2.0)

        rand_a = np.random.random(mask_a.shape) < P_A_TO_B_PER_STEP
        rand_b = np.random.random(mask_b.shape) < P_B_TO_A_PER_STEP

        switch_a_to_b = mask_a & rand_a
        switch_b_to_a = mask_b & rand_b

        num_switches = np.sum(switch_a_to_b) + np.sum(switch_b_to_a)
        self.diagnostics["Death_PhenotypeSwitch"] += int(num_switches)

        self.State[:, :, :, CELL_TYPE][switch_a_to_b] = 2.0
        self.State[:, :, :, CELL_TYPE][switch_b_to_a] = 1.0

        # -----------------------------------------------------------------
        # 3. ACTION TIMERS & PROLIFERATION ELIGIBILITY
        # -----------------------------------------------------------------
        self.State[:, :, :, STEP_COUNTER][active_mask] += 1.0

        eligible_a = (self.State[:, :, :, CELL_TYPE] == 1.0) & (self.State[:, :, :, STEP_COUNTER] % PROLIF_STEPS_A == 0)
        eligible_b = (self.State[:, :, :, CELL_TYPE] == 2.0) & (self.State[:, :, :, STEP_COUNTER] % BASE_PROLIF_STEPS_B == 0)
        eligible_c = (self.State[:, :, :, CELL_TYPE] == 3.0) & (self.State[:, :, :, STEP_COUNTER] % PROLIF_STEPS_C == 0)
        eligible_e = (self.State[:, :, :, CELL_TYPE] == 5.0) & (self.State[:, :, :, STEP_COUNTER] % PROLIF_STEPS_E == 0)

        waiting_mask = (self.State[:, :, :, WAITING_DIV] == 1.0) | eligible_a | eligible_b | eligible_c | eligible_e
        self.State[:, :, :, WAITING_DIV][waiting_mask] = 1.0

        # -----------------------------------------------------------------
        # 4. CONVOLUTION & PROLIFERATION GATING
        # -----------------------------------------------------------------
        occupied_grid = self.State[:, :, :, OCCUPIED].astype(np.uint8)
        neighbor_counts = convolve(occupied_grid, KERNEL_26, mode='constant', cval=0)
        empty_neighbors = 26 - neighbor_counts

        attempt_mask = (self.State[:, :, :, WAITING_DIV] == 1.0) & active_mask
        no_space_mask = attempt_mask & (empty_neighbors == 0)
        self.diagnostics["Prolif_NoEmptyNeighbors"] += int(np.sum(no_space_mask))

        valid_attempt = attempt_mask & (empty_neighbors > 0)
        local_occ = (26 - empty_neighbors) / 26.0
        p_success = P_PROLIF_BASE * ((1.0 - local_occ) ** GAMMA)

        rand_draws = np.random.random(occupied_grid.shape)
        successful_divs = valid_attempt & (rand_draws <= p_success)
        failed_gate = valid_attempt & (rand_draws > p_success)

        self.diagnostics["Prolif_ProbGateFailed"] += int(np.sum(failed_gate))
        self.diagnostics["Prolif_Succeeded"] += int(np.sum(successful_divs))

        self.State[:, :, :, WAITING_DIV][successful_divs] = 0.0

        if np.any(successful_divs):
            parent_coords = np.argwhere(successful_divs)
            for z, y, x in parent_coords:
                z_m, z_p = max(0, z-1), min(GRID_SIZE, z+2)
                y_m, y_p = max(0, y-1), min(GRID_SIZE, y+2)
                x_m, x_p = max(0, x-1), min(GRID_SIZE, x+2)

                sub_occ = self.State[z_m:z_p, y_m:y_p, x_m:x_p, OCCUPIED]
                empties = np.argwhere(sub_occ == 0.0)

                if len(empties) > 0:
                    dz, dy, dx = empties[np.random.choice(len(empties))]
                    nz, ny, nx = z_m + dz, y_m + dy, x_m + dx

                    p_type = self.State[z, y, x, CELL_TYPE]
                    self.State[nz, ny, nx, OCCUPIED] = 1.0
                    self.State[nz, ny, nx, CELL_TYPE] = p_type
                    self.State[nz, ny, nx, STEP_COUNTER] = 0.0

                    if p_type in [1.0, 2.0]:
                        d_lifespan = max(1.0, float(np.random.normal(MU_LIFESPAN_STEPS, SIGMA_LIFESPAN_STEPS)))
                        self.State[nz, ny, nx, REMAINING_LIFESPAN] = d_lifespan

        # -----------------------------------------------------------------
        # 5. MIGRATION & BOUNDARY REMOVAL
        # -----------------------------------------------------------------
        mig_a = (self.State[:, :, :, CELL_TYPE] == 1.0) & (self.State[:, :, :, STEP_COUNTER] % MIGRATE_STEPS_A == 0) & (self.State[:, :, :, WAITING_DIV] == 0.0)
        mig_b = (self.State[:, :, :, CELL_TYPE] == 2.0) & (self.State[:, :, :, STEP_COUNTER] % BASE_MIGRATE_STEPS_B == 0) & (self.State[:, :, :, WAITING_DIV] == 0.0)
        mig_c = (self.State[:, :, :, CELL_TYPE] == 3.0) & (self.State[:, :, :, STEP_COUNTER] % MIGRATE_STEPS_C == 0) & (self.State[:, :, :, WAITING_DIV] == 0.0)
        mig_d = (self.State[:, :, :, CELL_TYPE] == 4.0) & (self.State[:, :, :, STEP_COUNTER] % MIGRATE_STEPS_D == 0)
        mig_e = (self.State[:, :, :, CELL_TYPE] == 5.0) & (self.State[:, :, :, STEP_COUNTER] % MIGRATE_STEPS_E == 0)

        mig_mask = (mig_a | mig_b | mig_c | mig_d | mig_e) & (self.State[:, :, :, OCCUPIED] == 1.0)

        mig_coords = np.argwhere(mig_mask)
        for z, y, x in mig_coords:
            z_m, z_p = max(0, z-1), min(GRID_SIZE, z+2)
            y_m, y_p = max(0, y-1), min(GRID_SIZE, y+2)
            x_m, x_p = max(0, x-1), min(GRID_SIZE, x+2)

            sub_occ = self.State[z_m:z_p, y_m:y_p, x_m:x_p, OCCUPIED]
            empties = np.argwhere(sub_occ == 0.0)

            if len(empties) > 0:
                dz, dy, dx = empties[np.random.choice(len(empties))]
                nz, ny, nx = z_m + dz, y_m + dy, x_m + dx

                if (nz == 0 or ny == 0 or nx == 0 or nz == GRID_SIZE-1 or ny == GRID_SIZE-1 or nx == GRID_SIZE-1):
                    self.State[z, y, x, :] = 0.0
                else:
                    self.State[nz, ny, nx, :] = self.State[z, y, x, :]
                    self.State[z, y, x, :] = 0.0

        # -----------------------------------------------------------------
        # 6. CHEMOKINE FIELD DECAY
        # -----------------------------------------------------------------
        self.cxcl12_grid *= DECAY_FACTOR

    def get_counts(self):
        active = self.State[:, :, :, OCCUPIED] == 1.0
        a_count = np.sum(active & (self.State[:, :, :, CELL_TYPE] == 1.0))
        b_count = np.sum(active & (self.State[:, :, :, CELL_TYPE] == 2.0))
        c_count = np.sum(active & (self.State[:, :, :, CELL_TYPE] == 3.0))
        
        tcyto_mask = active & (self.State[:, :, :, CELL_TYPE] == 4.0)
        d_naive = np.sum(tcyto_mask & (self.State[:, :, :, SUB_STATE] == 0.0))
        d_act   = np.sum(tcyto_mask & (self.State[:, :, :, SUB_STATE] == 1.0))
        d_exh   = np.sum(tcyto_mask & (self.State[:, :, :, SUB_STATE] == 2.0))
        
        e_count = np.sum(active & (self.State[:, :, :, CELL_TYPE] == 5.0))
        return int(a_count), int(b_count), int(c_count), int(d_naive), int(d_act), int(d_exh), int(e_count)