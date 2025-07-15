import gymnasium as gym
from gymnasium import spaces
import numpy as np
from parameters import *
from channel_model_3gpp import caculate_offloading_rate


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
        super(UAVMECEnv, self).__init__()

        self.num_ues = num_ues
        self.num_uavs = num_uavs
        self.max_latency = max_latency
        self.episode_length = episode_length
        self.fmax_uav = fmax_uav
        self.t = 0

        # Propulsion power constant (simplified fixed value)
        self.p_propulsion_uav = 177  # W (paper dùng 177W)

        # Positions
        self.ue_positions = np.zeros((num_ues, 2))
        self.uav_positions = np.zeros((num_uavs, 2))

        # Tasks
        self.task_data_size = np.zeros(num_ues)
        self.task_cpu_cycles = np.zeros(num_ues)

        # UE computation capacity
        self.ue_computation_capacity = np.random.uniform(FMIN_UE, FMAX_UE, size=num_ues)

        self.offloading_decision = np.zeros(self.num_ues, dtype=int)
        self.transmission_power = np.zeros(self.num_ues)
        self.total_system_power = 0.0

        # === Action space: (offload decision, tx power level) ===
        # 0 = local, 1→N = UAV id | Lp power levels
        self.action_space = spaces.MultiDiscrete(
            [self.num_uavs + 1, NUM_TX_POWER_LEVELS]
        )

        # === State space: Z(t), P_TR(t), P_SYS(t) ===
        # Z(t): offload decision vector size K
        # P_TR(t): transmission power vector size K
        # P_SYS(t): system power scalar
        obs_low = np.zeros(self.num_ues * 2 + 1, dtype=np.float32)
        obs_high = np.concatenate(
            [
                np.full(self.num_ues, self.num_uavs, dtype=np.float32),
                np.full(self.num_ues, MAX_TX_POWER_W, dtype=np.float32),
                [1e4],
            ]
        ).astype(np.float32)
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # === Khởi tạo biến môi trường ===
        self.offloading_decision = np.zeros(self.num_ues, dtype=int)  # Z(t)
        self.transmission_power = np.zeros(self.num_ues)  # P_TR(t)
        self.total_system_power = 0.0  # P_SYS(t)

    def _init_uav_positions(self):
        angles = np.linspace(0, 2 * np.pi, self.num_uavs, endpoint=False)
        self.uav_positions = np.column_stack(
            (
                AREA_SIZE / 2 + UAV_RADIUS * np.cos(angles),
                AREA_SIZE / 2 + UAV_RADIUS * np.sin(angles),
            )
        )

    # def _generate_task(self):
    #     D_raw = np.random.uniform(0, 1, size=self.num_ues)
    #     self.task_data_size = D_raw / np.sum(D_raw) * self.max_task_data_sum

    #     F_raw = np.random.uniform(0, 1, size=self.num_ues)
    #     self.task_cpu_cycles = F_raw / np.sum(F_raw) * self.max_task_cpu_cycle_sum

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
        dx = speeds * np.cos(angles)
        dy = speeds * np.sin(angles)

        self.ue_positions += np.column_stack((dx, dy))
        self.ue_positions = np.clip(self.ue_positions, 0, AREA_SIZE)

    def _update_uav_positions(self):
        angles = (
            np.linspace(0, 2 * np.pi, self.num_uavs, endpoint=False)
            + self.t * UAV_SPEED / UAV_RADIUS
        )
        self.uav_positions = np.column_stack(
            (
                AREA_SIZE / 2 + UAV_RADIUS * np.cos(angles),
                AREA_SIZE / 2 + UAV_RADIUS * np.sin(angles),
            )
        )

    def _get_obs(self):
        """
        Trả về observation cho toàn bộ UE trong time slot
        """
        return np.concatenate(
            [
                self.offloading_decision,
                self.transmission_power,
                [self.total_system_power],
            ]
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ue_positions = np.random.uniform(0, AREA_SIZE, (self.num_ues, 2))
        self._init_uav_positions()
        self.task_data_size.fill(0.0)
        self.task_cpu_cycles.fill(0.0)
        self.t = 0
        self.offloading_decision = np.zeros(self.num_ues, dtype=int)
        self.transmission_power = np.zeros(self.num_ues)
        self.total_system_power = 0.0
        return self._get_obs(), {}

    def step(self, actions):
        self.t += 1

        violations_count = 0

        total_power_ue = 0.0
        total_power_uav = 0.0

        total_allocated_cpu_cycles = np.zeros(self.num_uavs)
        total_connected_ues = np.zeros(self.num_uavs)
        total_energy_consumption = 0.0

        self._generate_task()

        for i in range(self.num_ues):
            action = actions[i]
            if action[0] == 0:
                # Local processing
                self.offloading_decision[i] = 0
                self.transmission_power[i] = 0.0

                excute_time = self.task_cpu_cycles[i] / self.ue_computation_capacity[i]
                computation_power = KAPPA * self.ue_computation_capacity[i] ** NU
                if np.round(excute_time, 2) > self.max_latency:
                    violations_count += 1
                    print(
                        f"Violation: UE {i} local execution exceeds max latency {excute_time:.2f}s"
                    )

                total_power_ue += computation_power

                energy_consumption = excute_time * computation_power
                total_energy_consumption += energy_consumption
                # print(
                #     f"task data size: {self.task_data_size[i]:.2f} bits, "
                #     f"task CPU cycles: {self.task_cpu_cycles[i]:.2f} cycles"
                # )
                # print(
                #     f"UE computation capacity: {self.ue_computation_capacity[i]:.2f} cycles/s"
                # )
                # print(
                #     f"UE {i} local execution: {excute_time:.2f}s, power: {computation_power:.2f}W"
                # )
            else:
                # Offload to UAV
                self.offloading_decision[i] = action[0]
                self.transmission_power[i] = TX_POWER_LEVELS[action[1]]
                uav_id = action[0] - 1
                uav_position = self.uav_positions[uav_id]
                ue_position = self.ue_positions[i]
                offloading_rate = caculate_offloading_rate(
                    ue_position, uav_position, self.transmission_power[i], UAV_HEIGHT
                )
                tranmission_time = self.task_data_size[i] / offloading_rate

                if tranmission_time > self.max_latency:
                    violations_count += 1

                allocated_cpu_cycles = self.task_cpu_cycles[i] / (
                    TIME_SLOT_DURATION - tranmission_time
                )
                excute_time = self.task_cpu_cycles[i] / allocated_cpu_cycles
                computation_power = S_J * allocated_cpu_cycles**OMEGA_J
                total_power_uav += computation_power

                if np.round(tranmission_time + excute_time, 2) > self.max_latency:
                    violations_count += 1
                    print(
                        f"Violation: UE {i} offloading to UAV {uav_id + 1} exceeds max latency {self.max_latency:.2f}s"
                    )

                total_power_ue += self.transmission_power[i]

                total_allocated_cpu_cycles[uav_id] += allocated_cpu_cycles
                total_connected_ues[uav_id] += 1

                energy_consumption = (
                    tranmission_time * self.transmission_power[i]
                    + excute_time * computation_power
                )
                total_energy_consumption += energy_consumption

                # print(
                #     f"task data size: {self.task_data_size[i]:.2f} bits, "
                #     f"task CPU cycles: {self.task_cpu_cycles[i]:.2f} cycles"
                # )
                # print(
                #     f"UE {i} offloading to UAV {uav_id + 1}: "
                #     f"transmission time: {tranmission_time:.2f}s, "
                #     f"execution time: {excute_time:.2f}s, "
                #     f"transmission power: {self.transmission_power[i]:.2f}W, "
                #     f"computation power: {computation_power:.2f}W"
                # )

        for uav_id in range(self.num_uavs):
            if (
                total_connected_ues[uav_id] > CMAX_UE_PER_UAV
                or total_allocated_cpu_cycles[uav_id] > self.fmax_uav
            ):
                violations_count += 1
                print(
                    f"Violation: UAV {uav_id + 1} exceeds max constraints "
                    f"({total_connected_ues[uav_id]} UEs, "
                    f"{total_allocated_cpu_cycles[uav_id]:.2f} cycles/s)"
                )

        reward = 1 / (
            ZETA * total_power_ue
            + ETA * total_power_uav
            + violations_count * REWARD_PENALTY
        )

        terminated = self.t >= self.episode_length
        truncated = False

        # Cập nhật tổng công suất hệ thống
        self.total_system_power = total_power_ue + total_power_uav

        self._update_ue_positions()
        self._update_uav_positions()

        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            {
                "violations_count": violations_count,
                "total_power_ue": total_power_ue,
                "total_power_uav": total_power_uav,
                "total_system_power": self.total_system_power,
                "total_energy_consumption": total_energy_consumption,
            },
        )

    def render(self):
        pass

    def greedy_offloading(self):
        actions = []
        total_connected_ues = np.zeros(self.num_uavs)

        for i in range(self.num_ues):
            ue_pos = self.ue_positions[i]
            distances = np.linalg.norm(self.uav_positions - ue_pos, axis=1)
            sorted_uav_ids = np.argsort(distances)

            selected_uav = 0  # default local execute

            for uav_id in sorted_uav_ids:
                if total_connected_ues[uav_id] < CMAX_UE_PER_UAV:
                    selected_uav = uav_id + 1  # 1-based index (0 là local)
                    total_connected_ues[uav_id] += 1
                    break

            tx_power_level = np.random.randint(NUM_TX_POWER_LEVELS)
            actions.append([selected_uav, tx_power_level])

        return np.array(actions)

    def random_offloading(self):
        actions = []
        for _ in range(self.num_ues):
            offload_decision = np.random.randint(0, self.num_uavs + 1)
            tx_power_level = np.random.randint(NUM_TX_POWER_LEVELS)
            actions.append([offload_decision, tx_power_level])
        return np.array(actions)

    def local_execution(self):
        actions = []
        for _ in range(self.num_ues):
            actions.append([0, 0])  # local execute, không cần công suất truyền
        return np.array(actions)
