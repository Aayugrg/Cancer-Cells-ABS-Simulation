import random
import math
import numpy as np
from config import *

class CXCL12Field:
    def __init__(self):
        self.grid = np.zeros((MOLECULAR_GRID_SIZE, MOLECULAR_GRID_SIZE, MOLECULAR_GRID_SIZE), dtype=np.float32)
        self.substeps = 200
        self.dt_small = TIME_STEP_SEC / self.substeps
        computed_alpha = min(D_CXCL12 * self.dt_small / (10 ** 2), 0.15)
        self.alpha = min(computed_alpha, 0.15)

    def diffuse_and_decay(self):
        for _ in range(self.substeps):
            C = self.grid
            newC = C.copy()
            lap = (
                C[2:, 1:-1, 1:-1] + C[:-2, 1:-1, 1:-1] +
                C[1:-1, 2:, 1:-1] + C[1:-1, :-2, 1:-1] +
                C[1:-1, 1:-1, 2:] + C[1:-1, 1:-1, :-2] -
                6 * C[1:-1, 1:-1, 1:-1]
            )
            newC[1:-1, 1:-1, 1:-1] += self.alpha * lap
            self.grid = newC
        self.grid *= DECAY_FACTOR

    def secrete_from_cell(self, x, y, z, amount):
        bx, by, bz = x * 2, y * 2, z * 2
        self.grid[bz:bz+2, by:by+2, bx:bx+2] += amount / 8.0

    def get_cell_concentration(self, x, y, z):
        bx, by, bz = x * 2, y * 2, z * 2
        return float(np.mean(self.grid[bz:bz+2, by:by+2, bx:bx+2]))


class NaiveAgentA:
    def __init__(self, x, y, z):
        self.type = 'CXCR4-ve'
        self.x, self.y, self.z = x, y, z
        self.alive = True
        self.remaining_lifespan = self._draw_lifespan()
        self.divisions = 0
        self.step_counter = 0
        self.waiting_to_divide = False

    def _draw_lifespan(self):
        while True:
            u1, u2 = random.random(), random.random()
            z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            lifespan = int(MU_LIFESPAN_STEPS + SIGMA_LIFESPAN_STEPS * z)
            if lifespan > 0:
                return lifespan

    def step(self, grid, daily_diagnostics):
        if not self.alive: return []
        self.remaining_lifespan -= 1
        if self.remaining_lifespan <= 0:
            grid[self.z][self.y][self.x] = None
            self.alive = False
            daily_diagnostics["Death_NaturalLifespan"] += 1
            return []

        if random.random() < P_A_TO_B_PER_STEP:
            grid[self.z][self.y][self.x] = None
            new_cell = NaiveAgentB(self.x, self.y, self.z)
            grid[self.z][self.y][self.x] = new_cell
            daily_diagnostics["Death_PhenotypeSwitch"] += 1
            return [new_cell]

        self.step_counter += 1
        if self.step_counter % PROLIF_STEPS_A == 0:
            self.waiting_to_divide = True

        if self.waiting_to_divide:
            return self.proliferate(grid, daily_diagnostics)

        if self.step_counter % MIGRATE_STEPS_A == 0:
            self.migrate(grid)

        return [self]

    def proliferate(self, grid, daily_diagnostics):
        empties = self._find_empty_neighbors(grid)
        if not empties:
            daily_diagnostics["Prolif_NoEmptyNeighbors"] += 1
            return [self]

        local_occ = (26 - len(empties)) / 26.0
        p_success = P_PROLIF_BASE * ((1.0 - local_occ) ** GAMMA)

        if random.random() > p_success:
            daily_diagnostics["Prolif_ProbGateFailed"] += 1
            return [self]

        daily_diagnostics["Prolif_Succeeded"] += 1
        nx, ny, nz = random.choice(empties)
        daughter = NaiveAgentA(nx, ny, nz)
        grid[nz][ny][nx] = daughter
        self.divisions += 1
        self.waiting_to_divide = False
        return [self, daughter]

    def migrate(self, grid):
        empties = self._find_empty_neighbors(grid)
        if empties:
            nx, ny, nz = random.choice(empties)
            if self._is_edge(nx, ny, nz):
                grid[self.z][self.y][self.x] = None
                self.alive = False
                return
            grid[self.z][self.y][self.x] = None
            self.x, self.y, self.z = nx, ny, nz
            grid[nz][ny][nx] = self

    def _find_empty_neighbors(self, grid):
        empties = []
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                for dz in [-1,0,1]:
                    if dx == dy == dz == 0: continue
                    nx, ny, nz = self.x+dx, self.y+dy, self.z+dz
                    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and 0 <= nz < GRID_SIZE:
                        if grid[nz][ny][nx] is None:
                            empties.append((nx, ny, nz))
        return empties

    def _is_edge(self, x, y, z):
        return (x == 0 or y == 0 or z == 0 or x == GRID_SIZE-1 or y == GRID_SIZE-1 or z == GRID_SIZE-1)


class NaiveAgentB(NaiveAgentA):
    def __init__(self, x, y, z):
        super().__init__(x, y, z)
        self.type = 'CXCR4+ve'
        self.activated = False

    def step(self, grid, cxcl12_field, daily_diagnostics):
        if not self.alive: return []
        self.remaining_lifespan -= 1
        if self.remaining_lifespan <= 0:
            grid[self.z][self.y][self.x] = None
            self.alive = False
            daily_diagnostics["Death_NaturalLifespan"] += 1
            return []

        if random.random() < P_B_TO_A_PER_STEP:
            grid[self.z][self.y][self.x] = None
            new_cell = NaiveAgentA(self.x, self.y, self.z)
            grid[self.z][self.y][self.x] = new_cell
            daily_diagnostics["Death_PhenotypeSwitch"] += 1
            return [new_cell]

        conc = cxcl12_field.get_cell_concentration(self.x, self.y, self.z)
        self.activated = conc > KD_CXCL12

        self.step_counter += 1
        prolif_interval = ACT_PROLIF_STEPS_B if self.activated else BASE_PROLIF_STEPS_B
        mig_interval = ACT_MIGRATE_STEPS_B if self.activated else BASE_MIGRATE_STEPS_B

        if self.step_counter % prolif_interval == 0:
            self.waiting_to_divide = True

        if self.waiting_to_divide:
            return self.proliferate(grid, daily_diagnostics)

        if self.step_counter % mig_interval == 0:
            self.migrate(grid)

        return [self]


class NaiveAgentC:
    def __init__(self, x, y, z):
        self.type = 'Stromal'
        self.x, self.y, self.z = x, y, z
        self.alive = True
        self.step_counter = 0
        self.total_steps_lived = 0
        self.divisions = 0
        self.waiting_to_divide = False
        self.contact_steps = 0
        self.secretion_state = "baseline"
        self.senescent = False

    def step(self, grid, cxcl12_field):
        if not self.alive: return []
        self.total_steps_lived += 1

        if self.senescent:
            if random.random() < SENESCENT_DEATH_PER_STEP_C:
                grid[self.z][self.y][self.x] = None
                self.alive = False
                return []
            return [self]

        if self._cancer_in_neighborhood(grid):
            self.contact_steps += 1
        else:
            self.contact_steps = 0

        if self.contact_steps >= CONTACT_THRESHOLD_C:
            self.secretion_state = "high"

        amount = 4 if self.secretion_state == "high" else 1
        cxcl12_field.secrete_from_cell(self.x, self.y, self.z, amount)

        self.step_counter += 1
        if self.total_steps_lived < STOP_GROWTH_STEP_C:
            if self.step_counter % PROLIF_STEPS_C == 0:
                self.waiting_to_divide = True

        if self.waiting_to_divide:
            return self.proliferate(grid)

        if self.step_counter % MIGRATE_STEPS_C == 0:
            self.migrate(grid)

        return [self]

    def proliferate(self, grid):
        if self.divisions >= MAX_DIVISIONS_C:
            self.waiting_to_divide = False
            self.senescent = True
            return [self]

        empties = self._find_empty_neighbors(grid)
        if not empties:
            if self.total_steps_lived >= STOP_GROWTH_STEP_C:
                self.waiting_to_divide = False
            return [self]

        nx, ny, nz = random.choice(empties)
        daughter = NaiveAgentC(nx, ny, nz)
        daughter.total_steps_lived = self.total_steps_lived
        grid[nz][ny][nx] = daughter
        self.divisions += 1
        daughter.divisions = self.divisions
        self.waiting_to_divide = False
        return [self, daughter]

    def migrate(self, grid):
        empties = self._find_empty_neighbors(grid)
        if empties:
            nx, ny, nz = random.choice(empties)
            grid[self.z][self.y][self.x] = None
            self.x, self.y, self.z = nx, ny, nz
            grid[nz][ny][nx] = self

    def _cancer_in_neighborhood(self, grid):
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                for dz in [-1,0,1]:
                    if dx == dy == dz == 0: continue
                    nx, ny, nz = self.x+dx, self.y+dy, self.z+dz
                    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and 0 <= nz < GRID_SIZE:
                        cell = grid[nz][ny][nx]
                        if cell and cell.type in ['CXCR4-ve', 'CXCR4+ve']:
                            return True
        return False

    def _find_empty_neighbors(self, grid):
        empties = []
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                for dz in [-1,0,1]:
                    if dx == dy == dz == 0: continue
                    nx, ny, nz = self.x+dx, self.y+dy, self.z+dz
                    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and 0 <= nz < GRID_SIZE:
                        if grid[nz][ny][nx] is None:
                            empties.append((nx, ny, nz))
        return empties


class NaiveAgentD:
    def __init__(self, x, y, z):
        self.type = 'Tcyto'
        self.x, self.y, self.z = x, y, z
        self.alive = True
        self.state = "naive"
        self.divisions = 0
        self.step_counter = 0
        self.senescent = False
        self.contact_steps = 0
        self.activation_step = None
        self.first_division_delay = None
        self.last_division_step = None
        self.p_kill_exhausted = None
        self.p_prolif_exhausted = None
        self.p_migr_exhausted = None

    def step(self, grid):
        if not self.alive: return []
        self.step_counter += 1

        if self.senescent:
            if random.random() < SENESCENT_DEATH_PER_STEP_D:
                grid[self.z][self.y][self.x] = None
                self.alive = False
                return []
            return [self]

        if self.state == "naive":
            if random.random() < NAIVE_DEATH_PER_STEP_D:
                grid[self.z][self.y][self.x] = None
                self.alive = False
                return []
            self._try_activation(grid)
            if self.state == "naive":
                if self.step_counter % MIGRATE_STEPS_D == 0:
                    self._migrate_random(grid)
                return [self]

        if self.state == "activated":
            self._check_exhaustion(grid)

        killed = self._attempt_kill(grid)

        if not killed and self.step_counter % MIGRATE_STEPS_D == 0:
            self._migrate(grid)

        offspring = self._handle_proliferation(grid)
        return [self, offspring] if offspring else [self]

    def _try_activation(self, grid):
        cancer_count = self._count_cancer_neighbors(grid)
        if cancer_count > 0: self.contact_steps += 1
        else:
            self.contact_steps = 0
            return

        p = 1.0 if self.contact_steps > CONTACT_ACT_THRESHOLD_D else (P_ACT_LOW_D if cancer_count <= 3 else P_ACT_HIGH_D)
        if random.random() < p:
            self.state = "activated"
            self.activation_step = self.step_counter
            self.first_division_delay = random.randint(FIRST_DIV_MIN_STEPS_D, FIRST_DIV_MAX_STEPS_D)
            self.last_division_step = self.step_counter

    def _check_exhaustion(self, grid):
        treg_count = sum(1 for dx in [-1,0,1] for dy in [-1,0,1] for dz in [-1,0,1]
                         if not (dx==dy==dz==0) and 0 <= self.x+dx < GRID_SIZE and 0 <= self.y+dy < GRID_SIZE and 0 <= self.z+dz < GRID_SIZE
                         and grid[self.z+dz][self.y+dy][self.x+dx] and grid[self.z+dz][self.y+dy][self.x+dx].type == "Treg")

        if treg_count == 0: return
        exhausted = True if treg_count > TREG_EXHAUST_THRESHOLD_D else (random.random() < P_EXHAUSTION_D)

        if exhausted:
            if DIV_LIMIT_ACTIVATED_D - self.divisions < 2:
                grid[self.z][self.y][self.x] = None
                self.alive = False
                return
            self.state = "exhausted"
            self.p_kill_exhausted = random.uniform(0.1, 0.4)
            self.p_prolif_exhausted = random.uniform(0.1, 0.4)
            self.p_migr_exhausted = random.uniform(0.1, 0.4)
            self.last_division_step = self.step_counter

    def _attempt_kill(self, grid):
        if self.state not in ["activated", "exhausted"]: return False
        c_neighbors = [grid[self.z+dz][self.y+dy][self.x+dx] for dx in [-1,0,1] for dy in [-1,0,1] for dz in [-1,0,1]
                       if not (dx==dy==dz==0) and 0 <= self.x+dx < GRID_SIZE and 0 <= self.y+dy < GRID_SIZE and 0 <= self.z+dz < GRID_SIZE
                       and grid[self.z+dz][self.y+dy][self.x+dx] and grid[self.z+dz][self.y+dy][self.x+dx].type in ["CXCR4-ve", "CXCR4+ve"]]

        if not c_neighbors: return False
        p = P_KILL_ACTIVATED_D if self.state == "activated" else self.p_kill_exhausted
        if random.random() < p:
            victim = random.choice(c_neighbors)
            grid[victim.z][victim.y][victim.x] = None
            victim.alive = False
            return True
        return False

    def _migrate_random(self, grid):
        empties = self._find_empty_neighbors(grid)
        if empties:
            nx, ny, nz = random.choice(empties)
            grid[self.z][self.y][self.x] = None
            self.x, self.y, self.z = nx, ny, nz
            grid[nz][ny][nx] = self

    def _migrate(self, grid):
        if self.state == "exhausted" and random.random() > self.p_migr_exhausted: return
        empties = self._find_empty_neighbors(grid)
        if empties:
            nx, ny, nz = random.choice(empties)
            grid[self.z][self.y][self.x] = None
            self.x, self.y, self.z = nx, ny, nz
            grid[nz][ny][nx] = self

    def _handle_proliferation(self, grid):
        if self.state == "naive": return None
        limit = DIV_LIMIT_ACTIVATED_D if self.state == "activated" else DIV_LIMIT_EXHAUSTED_D
        interval = ACT_DIV_INTERVAL_D if self.state == "activated" else EXH_DIV_INTERVAL_D

        if self.divisions >= limit:
            self.senescent = True
            return None

        steps_since_act = self.step_counter - self.activation_step
        if self.divisions == 0 and steps_since_act < self.first_division_delay: return None
        if self.divisions > 0 and self.step_counter - self.last_division_step < interval: return None

        empties = self._find_empty_neighbors(grid)
        if empties:
            nx, ny, nz = random.choice(empties)
            daughter = NaiveAgentD(nx, ny, nz)
            grid[nz][ny][nx] = daughter
            self.divisions += 1
            self.last_division_step = self.step_counter
            return daughter
        return None

    def _count_cancer_neighbors(self, grid):
        return sum(1 for dx in [-1,0,1] for dy in [-1,0,1] for dz in [-1,0,1]
                   if not (dx==dy==dz==0) and 0 <= self.x+dx < GRID_SIZE and 0 <= self.y+dy < GRID_SIZE and 0 <= self.z+dz < GRID_SIZE
                   and grid[self.z+dz][self.y+dy][self.x+dx] and grid[self.z+dz][self.y+dy][self.x+dx].type in ["CXCR4-ve", "CXCR4+ve"])

    def _find_empty_neighbors(self, grid):
        empties = []
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                for dz in [-1,0,1]:
                    if dx == dy == dz == 0: continue
                    nx, ny, nz = self.x+dx, self.y+dy, self.z+dz
                    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and 0 <= nz < GRID_SIZE:
                        if grid[nz][ny][nx] is None: empties.append((nx, ny, nz))
        return empties


class NaiveAgentE:
    def __init__(self, x, y, z):
        self.type = "Treg"
        self.x, self.y, self.z = x, y, z
        self.alive = True
        self.step_counter = 0
        self.divisions = 0
        self.senescent = False

    def step(self, grid, cxcl12_field):
        if not self.alive: return []
        self.step_counter += 1

        if random.random() < TREG_DEATH_PER_STEP_E:
            grid[self.z][self.y][self.x] = None
            self.alive = False
            return []

        if self.senescent:
            if random.random() < SENESCENT_DEATH_PER_STEP_E:
                grid[self.z][self.y][self.x] = None
                self.alive = False
                return []
            return [self]

        if self.step_counter % MIGRATE_STEPS_E == 0:
            self._migrate(grid, cxcl12_field)

        offspring = None
        if self.step_counter % PROLIF_STEPS_E == 0:
            offspring = self._proliferate(grid)

        return [self, offspring] if offspring else [self]

    def _migrate(self, grid, cxcl12_field):
        empties = self._find_empty_neighbors(grid)
        if not empties: return
        best_cells, best_val = [], -1
        for ex, ey, ez in empties:
            c = cxcl12_field.get_cell_concentration(ex, ey, ez)
            if c > best_val:
                best_val = c
                best_cells = [(ex, ey, ez)]
            elif c == best_val:
                best_cells.append((ex, ey, ez))

        nx, ny, nz = random.choice(best_cells if best_val > 0 else empties)
        grid[self.z][self.y][self.x] = None
        self.x, self.y, self.z = nx, ny, nz
        grid[nz][ny][nx] = self

    def _proliferate(self, grid):
        if self._count_cancer_neighbors(grid) == 0: return None
        if random.random() > 0.35: return None
        if self.divisions >= MAX_DIVISIONS_E:
            self.senescent = True
            return None

        empties = self._find_empty_neighbors(grid)
        if empties:
            nx, ny, nz = random.choice(empties)
            daughter = NaiveAgentE(nx, ny, nz)
            grid[nz][ny][nx] = daughter
            self.divisions += 1
            return daughter
        return None

    def _count_cancer_neighbors(self, grid):
        return sum(1 for dx in [-1,0,1] for dy in [-1,0,1] for dz in [-1,0,1]
                   if not (dx==dy==dz==0) and 0 <= self.x+dx < GRID_SIZE and 0 <= self.y+dy < GRID_SIZE and 0 <= self.z+dz < GRID_SIZE
                   and grid[self.z+dz][self.y+dy][self.x+dx] and grid[self.z+dz][self.y+dy][self.x+dx].type in ["CXCR4-ve", "CXCR4+ve"])

    def _find_empty_neighbors(self, grid):
        empties = []
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                for dz in [-1,0,1]:
                    if dx == dy == dz == 0: continue
                    nx, ny, nz = self.x+dx, self.y+dy, self.z+dz
                    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and 0 <= nz < GRID_SIZE:
                        if grid[nz][ny][nx] is None: empties.append((nx, ny, nz))
        return empties


class NaiveSimulation:
    def __init__(self):
        self.grid = [[[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.agents = []
        self.cxcl12_field = CXCL12Field()
        self.diagnostics = {
            "Prolif_Succeeded": 0, "Prolif_NoEmptyNeighbors": 0,
            "Prolif_ProbGateFailed": 0, "Death_NaturalLifespan": 0,
            "Death_PhenotypeSwitch": 0
        }

    def seed(self, num_a, num_b, num_c, num_d, num_e):
        total_cells = num_a + num_b + num_c + num_d + num_e
        indices = np.random.choice(GRID_SIZE**3, size=total_cells, replace=False)
        for i, idx in enumerate(indices):
            z = idx // (GRID_SIZE**2)
            rem = idx % (GRID_SIZE**2)
            y = rem // GRID_SIZE
            x = rem % GRID_SIZE
            if i < num_a: cell = NaiveAgentA(x, y, z)
            elif i < num_a + num_b: cell = NaiveAgentB(x, y, z)
            elif i < num_a + num_b + num_c: cell = NaiveAgentC(x, y, z)
            elif i < num_a + num_b + num_c + num_d: cell = NaiveAgentD(x, y, z)
            else: cell = NaiveAgentE(x, y, z)
            self.grid[z][y][x] = cell
            self.agents.append(cell)

    def step(self):
        random.shuffle(self.agents)
        next_agents = []
        for agent in self.agents:
            if agent.alive:
                # Check NaiveAgentB BEFORE NaiveAgentA because AgentB inherits from AgentA!
                if isinstance(agent, NaiveAgentB):
                    res = agent.step(self.grid, self.cxcl12_field, self.diagnostics)
                elif isinstance(agent, NaiveAgentA):
                    res = agent.step(self.grid, self.diagnostics)
                elif isinstance(agent, NaiveAgentC):
                    res = agent.step(self.grid, self.cxcl12_field)
                elif isinstance(agent, NaiveAgentD):
                    res = agent.step(self.grid)
                elif isinstance(agent, NaiveAgentE):
                    res = agent.step(self.grid, self.cxcl12_field)
                
                next_agents.extend(res)
        self.agents = next_agents
        self.cxcl12_field.diffuse_and_decay()

    def get_counts(self):
        a_count = sum(1 for a in self.agents if a.alive and a.type == 'CXCR4-ve')
        b_count = sum(1 for a in self.agents if a.alive and a.type == 'CXCR4+ve')
        c_count = sum(1 for a in self.agents if a.alive and a.type == 'Stromal')
        d_naive = sum(1 for a in self.agents if a.alive and a.type == 'Tcyto' and a.state == 'naive')
        d_act = sum(1 for a in self.agents if a.alive and a.type == 'Tcyto' and a.state == 'activated')
        d_exh = sum(1 for a in self.agents if a.alive and a.type == 'Tcyto' and a.state == 'exhausted')
        e_count = sum(1 for a in self.agents if a.alive and a.type == 'Treg')
        return a_count, b_count, c_count, d_naive, d_act, d_exh, e_count