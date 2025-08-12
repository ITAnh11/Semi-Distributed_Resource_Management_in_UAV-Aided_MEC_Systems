import gymnasium as gym
from gymnasium import spaces
import numpy as np
from parameters import *
from channel_model_3gpp import caculate_offloading_rate
import random


class UAVMECEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_ues=100,
        num_uavs=10,
        max_latency=1.0,
        episode_length=EPISODE_LENGTH,
        fmax_uav=FMAX_UAV,
    ):
        super().__init__()

        self.num_ues = num_ues
        self.num_uavs = num_uavs
        self.max_latency = max_latency
        self.episode_length = episode_length
        self.fmax_uav = fmax_uav
        self.t = 0
        self.p_propulsion_uav = 177  # W

        self.ue_positions = np.zeros((num_ues, 2))
        self.uav_positions = np.zeros((num_uavs, 2))

        self.task_data_size = np.zeros(num_ues)
        self.task_cpu_cycles = np.zeros(num_ues)

        self.ue_computation_capacity = np.random.uniform(FMIN_UE, FMAX_UE, size=num_ues)

        self.offloading_decision = np.zeros(num_ues, dtype=int)
        self.transmission_power = np.zeros(num_ues)
        self.total_system_power = 0.0

        # action_id ∈ [0, (num_uavs+1) * NUM_TX_POWER_LEVELS - 1]
        # offload = action_id // NUM_TX_POWER_LEVELS ∈ [0, num_uavs]
        # power_idx = action_id % NUM_TX_POWER_LEVELS ∈ [0, NUM_TX_POWER_LEVELS-1]
        self.total_actions = (self.num_uavs + 1) * NUM_TX_POWER_LEVELS
        self.action_space = spaces.Discrete(self.total_actions)

        obs_low = np.zeros(num_ues * 2 + 1, dtype=np.float32)
        obs_high = np.concatenate(
            [
                np.full(num_ues, self.num_uavs, dtype=np.float32),
                np.full(num_ues, MAX_TX_POWER_W, dtype=np.float32),
                [1e2],
            ]
        )
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

    def _init_uav_positions(self):
        angles = np.linspace(0, 2 * np.pi, self.num_uavs, endpoint=False)
        self.uav_positions = np.column_stack(
            [
                AREA_SIZE / 2 + UAV_RADIUS * np.cos(angles),
                AREA_SIZE / 2 + UAV_RADIUS * np.sin(angles),
            ]
        )

    def _generate_task(self):
        self.task_data_size = np.random.uniform(
            TASK_DATA_MIN, TASK_DATA_MAX, size=self.num_ues
        )
        self.task_cpu_cycles = np.random.uniform(
            TASK_CPU_MIN, TASK_CPU_MAX, size=self.num_ues
        )

    def _update_ue_positions(self):
        speeds = np.random.uniform(UE_SPEED_MIN, UE_SPEED_MAX, self.num_ues)
        angles = np.random.uniform(UE_MOVE_ANGLE_MIN, UE_MOVE_ANGLE_MAX, self.num_ues)
        movement = np.column_stack((speeds * np.cos(angles), speeds * np.sin(angles)))
        self.ue_positions = np.clip(self.ue_positions + movement, 0, AREA_SIZE)

    def _update_uav_positions(self):
        angles = (
            np.linspace(0, 2 * np.pi, self.num_uavs, endpoint=False)
            + self.t * UAV_SPEED / UAV_RADIUS
        )
        self.uav_positions = np.column_stack(
            [
                AREA_SIZE / 2 + UAV_RADIUS * np.cos(angles),
                AREA_SIZE / 2 + UAV_RADIUS * np.sin(angles),
            ]
        )

    def _get_obs(self):
        offload_norm = self.offloading_decision / self.num_uavs
        power_norm = self.transmission_power / MAX_TX_POWER_W
        power_sys_norm = np.array([self.total_system_power / 1e2])
        return np.concatenate([offload_norm, power_norm, power_sys_norm]).astype(
            np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)  # numpy
            random.seed(seed)  # random
            # torch.manual_seed(seed) # nếu có dùng torch trong env

        self.ue_positions = np.random.uniform(0, AREA_SIZE, (self.num_ues, 2))
        self._init_uav_positions()
        self.task_data_size.fill(0.0)
        self.task_cpu_cycles.fill(0.0)
        self.t = 0
        self.offloading_decision.fill(0)
        self.transmission_power.fill(0.0)
        self.total_system_power = 0.0
        return self._get_obs(), {}

    def step(self, actions):
        self.t += 1
        violations = 0
        total_power_ue = 0.0
        total_power_uav = 0.0
        total_energy = 0.0

        cpu_allocated = np.zeros(self.num_uavs)
        connected_ues = np.zeros(self.num_uavs)

        self._generate_task()

        for i, action_id in enumerate(actions):
            offload = int(action_id) // NUM_TX_POWER_LEVELS
            power_idx = int(action_id) % NUM_TX_POWER_LEVELS
            if offload == 0:
                exc_time = self.task_cpu_cycles[i] / self.ue_computation_capacity[i]
                comp_power = KAPPA * self.ue_computation_capacity[i] ** NU
                if exc_time > self.max_latency:
                    violations += 1
                total_power_ue += comp_power
                total_energy += exc_time * comp_power
            else:
                power = TX_POWER_LEVELS[power_idx]
                uav_id = offload - 1
                rate = caculate_offloading_rate(
                    self.ue_positions[i], self.uav_positions[uav_id], power, UAV_HEIGHT
                )
                tx_time = self.task_data_size[i] / rate
                if tx_time > self.max_latency:
                    violations += 1

                cpu_req = self.task_cpu_cycles[i] / max(
                    (TIME_SLOT_DURATION - tx_time), 1e-6
                )
                exc_time = self.task_cpu_cycles[i] / cpu_req
                comp_power = S_J * cpu_req**OMEGA_J
                if tx_time + exc_time > self.max_latency:
                    violations += 1

                total_power_ue += power
                total_power_uav += comp_power
                total_energy += tx_time * power + exc_time * comp_power

                cpu_allocated[uav_id] += cpu_req
                connected_ues[uav_id] += 1

            self.offloading_decision[i] = offload
            self.transmission_power[i] = (
                0.0 if offload == 0 else TX_POWER_LEVELS[power_idx]
            )

        for uid in range(self.num_uavs):
            if (
                connected_ues[uid] > CMAX_UE_PER_UAV
                or cpu_allocated[uid] > self.fmax_uav
            ):
                violations += 1

        reward = 1 / (
            ZETA * total_power_ue + ETA * total_power_uav + violations * REWARD_PENALTY
        )
        self.total_system_power = total_power_ue + total_power_uav

        self._update_ue_positions()
        self._update_uav_positions()

        return (
            self._get_obs(),
            reward,
            self.t >= self.episode_length,
            False,
            {
                "violations_count": violations,
                "total_power_ue": total_power_ue,
                "total_power_uav": total_power_uav,
                "total_system_power": self.total_system_power,
                "total_energy_consumption": total_energy,
            },
        )

    def render(self):
        pass

    def greedy_offloading(self):
        actions = []
        ue_count = np.zeros(self.num_uavs)
        for ue_pos in self.ue_positions:
            dists = np.linalg.norm(self.uav_positions - ue_pos, axis=1)
            sorted_ids = np.argsort(dists)
            offload = 0
            for uid in sorted_ids:
                if ue_count[uid] < CMAX_UE_PER_UAV:
                    offload = uid + 1
                    ue_count[uid] += 1
                    break
            power_idx = np.random.randint(NUM_TX_POWER_LEVELS)
            action_id = offload * NUM_TX_POWER_LEVELS + power_idx
            actions.append(action_id)
        return np.array(actions)

    def random_offloading(self):
        total_actions = (self.num_uavs + 1) * NUM_TX_POWER_LEVELS
        return np.random.randint(0, total_actions, self.num_ues)

    def local_execution(self):
        return np.zeros(
            self.num_ues, dtype=int
        )  # offload = 0, power = 0 → action_id = 0

    def get_state_ue(self, ue_id):
        offload_norm = self.offloading_decision[ue_id] / self.num_uavs
        power_norm = self.transmission_power[ue_id] / MAX_TX_POWER_W
        ue_position_norm = self.ue_positions[ue_id] / AREA_SIZE
        uav_positions_norm = self.uav_positions / AREA_SIZE
        power_sys_norm = self.total_system_power / 1e2
        return np.concatenate(
            [
                ue_position_norm,
                uav_positions_norm.flatten(),
                [offload_norm, power_norm, power_sys_norm],
            ]
        ).astype(np.float32)
