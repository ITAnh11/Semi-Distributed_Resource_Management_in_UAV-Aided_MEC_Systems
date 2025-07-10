import gymnasium as gym
from gymnasium import spaces
import numpy as np
from parameters import *
from utils_custom.chanel_model import *


class UAVMECEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        num_ues=NUM_UES_K,
        num_uavs=NUM_UAVS_N,
        max_latency_constraint=MAXIMAL_LATENCY_CONSTRAINT_TC,
        max_computation_capacity_uav=MAX_COMPUTATION_CAPACITY_UAV_F_MAX,
        total_cpu_cycles=TOTAL_CPU_CYCLES_F,
        total_data_size=TOTAL_DATA_SIZE_D,
    ):
        super(UAVMECEnv, self).__init__()

        # Parameters
        self.num_ues = num_ues  # 1,2,...,NUM_UES_K
        self.num_uavs = num_uavs  # 1,2,...,NUM_UAVS_N
        self.time_slot = 0
        self.max_latency_constraint = max_latency_constraint  # s
        self.max_computation_capacity_uav = max_computation_capacity_uav  # CPU cycles
        self.total_cpu_cycles = total_cpu_cycles  # CPU cycles
        self.total_data_size = total_data_size * 1e3  # Bytes

        # State: [UE offload target, UE transmit power level, total power]
        self.observation_space = spaces.Box(
            low=np.array(
                [0] * self.num_ues + [0.0] * self.num_ues + [0.0], dtype=np.float32
            ),
            high=np.array(
                [self.num_uavs] * self.num_ues
                + [MAXIMAL_TRANSMISSION_POWER_UES_P_TR_MAX] * self.num_ues
                + [1e5],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # Action: for each UE => [offload_target (0..num_uavs), power_level (0..Lp-1)]
        self.action_space = spaces.MultiDiscrete(
            [num_uavs + 1, NUMBER_TRANSMISSION_POWER_LEVELS_LP] * num_ues
        )

        # Task parameters per UE (randomized)
        self.data_size = np.zeros(num_ues, dtype=np.float32)
        self.cpu_cycles = np.zeros(num_ues, dtype=np.float32)

        self.parameter_server_pos = np.array(
            [RECTANGULAR_AREA_SIDE_LENGTH / 2, RECTANGULAR_AREA_SIDE_LENGTH / 2, 0]
        )

        # Position of UEs (x, y, z)
        self.ue_pos = np.zeros((num_ues, 3), dtype=np.float32)

        self.ue_computation_capacity = np.random.uniform(
            UE_COMPUTATION_CAPACITY_FI0_RANGE[0],
            UE_COMPUTATION_CAPACITY_FI0_RANGE[1],
            self.num_ues,
        )

        # Position of UAV(x, y, z)
        self.uav_pos = np.zeros((num_uavs, 3), dtype=np.float32)

        self.overall_transmission_power = 0
        self.overall_power_consumption_system = 0

        self.number_connected_of_uavs = np.zeros(
            self.num_uavs, dtype=np.int32
        )  # Count of UEs connected to each UAV

        self.uav_velocity = UAV_VELOCITY_V  # m/s

        self.propulsion_power_uav = (
            BLADE_PROFILE_POWER_PB
            * (1 + (3 * UAV_VELOCITY_V**2) / TIP_SPEED_BLADE_UT**2)
            + INDUCED_POWER_PI
            * (
                (1 + UAV_VELOCITY_V**4 / (4 * MEAN_INDUCED_SPEED_V0**4)) ** 0.5
                - UAV_VELOCITY_V**2 / (2 * MEAN_INDUCED_SPEED_V0**2)
            )
            ** 0.5
            + 0.5
            * FUSELAGE_DRAG_PROPORTION_F0
            * AIR_DENSITY_RHO
            * ROTOR_DISC_AREA_A
            * UAV_VELOCITY_V**3
        )  # (18)

    def _init_uav_positions(self):
        angle_step = 2 * np.pi / self.num_uavs
        for i in range(self.num_uavs):
            angle = i * angle_step
            x = (
                UAV_CIRCULAR_PATTERN_RADIUS_R * np.cos(angle)
                + self.parameter_server_pos[0]
            )
            y = (
                UAV_CIRCULAR_PATTERN_RADIUS_R * np.sin(angle)
                + self.parameter_server_pos[1]
            )
            self.uav_pos[i] = np.array([x, y, UAV_HEIGHT_H])

    def _init_ue_positions(self):
        # Initialize UE positions randomly within the rectangular area
        self.ue_pos = np.random.uniform(
            0, RECTANGULAR_AREA_SIDE_LENGTH, size=(self.num_ues, 2)
        )
        self.ue_pos = np.column_stack((self.ue_pos, np.zeros(self.num_ues)))

    def _update_ue_positions(self):
        # Update UE positions based on mobility constraints
        angles = np.random.uniform(
            UE_MOBILITY_ANGLE_CONSTRAINT[0],
            UE_MOBILITY_ANGLE_CONSTRAINT[1],
            self.num_ues,
        )

        velocities = np.random.uniform(
            UE_MOBILITY_VELOCITY_CONSTRAINT[0],
            UE_MOBILITY_VELOCITY_CONSTRAINT[1],
            self.num_ues,
        )

        for i in range(self.num_ues):
            x_curr, y_curr, z_curr = self.ue_pos[i]
            x_new = x_curr + velocities[i] * np.cos(angles[i]) * TIME_INTERVAL_PER_SLOT
            y_new = y_curr + velocities[i] * np.sin(angles[i]) * TIME_INTERVAL_PER_SLOT

            # Ensure UE stays within the rectangular area
            x_new = np.clip(x_new, 0, RECTANGULAR_AREA_SIDE_LENGTH)
            y_new = np.clip(y_new, 0, RECTANGULAR_AREA_SIDE_LENGTH)

            self.ue_pos[i] = np.array([x_new, y_new, z_curr])

    def _update_uav_positions(self):
        agular_velocity = UAV_VELOCITY_V / UAV_CIRCULAR_PATTERN_RADIUS_R
        for i, (x_curr, y_curr, z_curr) in enumerate(self.uav_pos):
            current_angle = np.arctan2(
                y_curr - self.parameter_server_pos[1],
                x_curr - self.parameter_server_pos[0],
            )
            new_angle = current_angle + agular_velocity * TIME_INTERVAL_PER_SLOT
            x_new = (
                UAV_CIRCULAR_PATTERN_RADIUS_R * np.cos(new_angle)
                + self.parameter_server_pos[0]
            )
            y_new = (
                UAV_CIRCULAR_PATTERN_RADIUS_R * np.sin(new_angle)
                + self.parameter_server_pos[1]
            )
            self.uav_pos[i] = np.array([x_new, y_new, UAV_HEIGHT_H])

    def random_with_total_sum(
        self, num_values, total_sum, low_ratio=0.5, high_ratio=1.5
    ):
        random_ratios = np.random.uniform(low_ratio, high_ratio, num_values)
        random_ratios /= np.sum(random_ratios)
        values = random_ratios * total_sum
        return values

    def _generate_task_data_and_cpu_demands(self):
        self.data_size = self.random_with_total_sum(
            self.num_ues,
            self.total_data_size,
            low_ratio=0.5,
            high_ratio=1.5,
        )
        self.cpu_cycles = self.random_with_total_sum(
            self.num_ues,
            self.total_cpu_cycles,
            low_ratio=0.5,
            high_ratio=1.5,
        )

    def _caculate_reward(self, action):
        count_violations = 0

        sum_transmission_power = 0

        overall_computation_resource_allocated_of_UAVs = np.zeros(
            self.num_uavs, dtype=np.float32
        )

        sum_computation_power_uavs = 0
        sum_local_computation_power = 0
        sum_propulsion_power_uavs = 0
        sum_time_processing_when_offload = 0
        sum_time_processing_when_local = 0

        transmission_power_ues = np.zeros(self.num_ues, dtype=np.float32)

        for ue_idx in range(self.num_ues):
            offload_target = action[ue_idx * 2]
            power_level = action[ue_idx * 2 + 1]

            if offload_target == 0:
                # Local execution
                local_computation_time = (
                    self.cpu_cycles[ue_idx] / self.ue_computation_capacity[ue_idx]
                )  # (14)

                KAPPA = 1e-27
                NU = 3
                local_computation_power = KAPPA * (
                    self.ue_computation_capacity[ue_idx] ** NU
                )  # (15)

                sum_local_computation_power += local_computation_power
                sum_time_processing_when_local += local_computation_time

                transmission_power_ues[ue_idx] = (
                    0  # No transmission power when local execution
                )
            else:
                self.number_connected_of_uavs[
                    offload_target - 1
                ] += 1  # -1 to match action space

                transmission_power = (
                    power_level / NUMBER_TRANSMISSION_POWER_LEVELS_LP
                ) * convert_dBm_to_W(MAXIMAL_TRANSMISSION_POWER_UES_P_TR_MAX)

                offload_rate = calculate_offloading_rate(
                    self.ue_pos[ue_idx],
                    self.uav_pos[offload_target - 1],  # -1 to match action space
                    transmission_power,
                )

                transmission_power_ues[ue_idx] = transmission_power
                sum_transmission_power += transmission_power  # (9)

                transmission_time_ue_uav = self.data_size[ue_idx] / offload_rate  # (6)
                allocated_cpu_cycles = self.cpu_cycles[ue_idx] / (
                    self.max_latency_constraint - transmission_time_ue_uav
                )  # (23)

                overall_computation_resource_allocated_of_UAVs[
                    offload_target - 1
                ] += allocated_cpu_cycles  # (11)

                task_execution_time = (
                    self.cpu_cycles[ue_idx] / allocated_cpu_cycles
                )  # (7)

                sum_time_processing_when_offload += (
                    task_execution_time + transmission_time_ue_uav
                )  # (8)

        if sum_time_processing_when_local > self.max_latency_constraint:
            count_violations += 1  # (21c)
        if sum_time_processing_when_offload > self.max_latency_constraint:
            count_violations += 1  # (21d)

        for uav_idx in range(self.num_uavs):
            if (
                overall_computation_resource_allocated_of_UAVs[uav_idx]
                > self.max_computation_capacity_uav
            ):
                count_violations += 1  # (12)

            if self.number_connected_of_uavs[uav_idx] > MAX_CONNECTED_UES_PER_UAV_C_MAX:
                count_violations += 1  # (13)

            sum_computation_power_uavs += (
                COMPUTATIONAL_CONSTANT_S
                * (overall_computation_resource_allocated_of_UAVs[uav_idx]) ** OMEGA
            )  # (10)

            sum_propulsion_power_uavs += self.propulsion_power_uav  # (19)

        P_TR_t = transmission_power_ues
        P_UE_t = sum_local_computation_power + sum_transmission_power
        P_UAV_t = sum_computation_power_uavs
        P_SYS_t = P_UE_t + P_UAV_t  # (20)

        print(
            f"""
            {transmission_power_ues},
            sum_transmission_power: {sum_transmission_power} W,
            sum_local_computation_power: {sum_local_computation_power} W,
            sum_computation_power_uavs: {sum_computation_power_uavs} W,
            sum_propulsion_power_uavs: {sum_propulsion_power_uavs} W,
            count_violations: {count_violations},
            P_SYS_t: {P_SYS_t} W"""
        )

        reward = 1 / (
            POWER_CONSUMPTION_WEIGHT_UAVS_ETA * P_UAV_t
            + POWER_CONSUMPTION_WEIGHT_UES_XI * P_UE_t
            + count_violations * AVERAGE_PENALTY_CONSTANT_EACH_VIOLATION_RP
        )  # (24)

        return reward, P_TR_t, P_SYS_t

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time_slot = 0
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        self._init_ue_positions()
        self._init_uav_positions()

        self.number_connected_of_uavs = np.zeros(self.num_uavs, dtype=np.int32)

        self.overall_transmission_power = 0
        self.overall_power_consumption_system = 0

        self.ue_computation_capacity = np.random.uniform(
            UE_COMPUTATION_CAPACITY_FI0_RANGE[0],
            UE_COMPUTATION_CAPACITY_FI0_RANGE[1],
            self.num_ues,
        )
        return obs, {}

    def step(self, action):
        self.time_slot += 1
        self._generate_task_data_and_cpu_demands()
        self.number_connected_of_uavs = np.zeros(self.num_uavs, dtype=np.int32)
        Z_t = [action[i * 2] for i in range(self.num_ues)]

        reward, P_TR_t, P_SYS_t = self._caculate_reward(action)
        obs = {
            "Z_t": Z_t,
            "P_TR_t": P_TR_t,
            "P_SYS_t": P_SYS_t,
        }

        done = self.time_slot >= NUM_TIME_SLOTS_T

        self._update_ue_positions()
        self._update_uav_positions()

        self.overall_transmission_power += P_TR_t
        self.overall_power_consumption_system += P_SYS_t

        return obs, reward, done, False, {}

    def render(self, file_name=None):
        if not file_name:
            print(
                f"Average system power consumption: {(self.overall_power_consumption_system) / self.time_slot:.2f} W"
            )

    def close(self):
        pass

    def get_action_when_random_execute(self):
        action = np.zeros(self.action_space.shape, dtype=np.int32)
        for ue_idx in range(self.num_ues):
            offload_target = np.random.randint(0, self.num_uavs + 1)
            power_level = np.random.randint(1, NUMBER_TRANSMISSION_POWER_LEVELS_LP + 1)
            action[ue_idx * 2] = offload_target
            action[ue_idx * 2 + 1] = power_level
        return action

    def get_action_when_greedy_offload(self):
        action = np.zeros(self.action_space.shape, dtype=np.int32)
        for ue_idx in range(self.num_ues):
            ue_pos = self.ue_pos[ue_idx]
            distances = []
            for uav_idx in range(self.num_uavs):
                uav_pos = self.uav_pos[uav_idx]
                d_3d = distance_3d(ue_pos, uav_pos, UAV_HEIGHT_H)
                distances.append(d_3d)
            sorted_indices = np.argsort(distances)
            for uav_idx in sorted_indices:
                if (
                    self.number_connected_of_uavs[uav_idx]
                    < MAX_CONNECTED_UES_PER_UAV_C_MAX
                ):
                    action[ue_idx * 2] = uav_idx + 1  # +1 to match action space
                    self.number_connected_of_uavs[uav_idx] += 1
                    break

            power_level = np.random.randint(
                1, NUMBER_TRANSMISSION_POWER_LEVELS_LP + 1
            )  # Random power level from 1 to LP
            action[ue_idx * 2 + 1] = power_level

        return action

    def get_action_when_local_execute(self):
        return np.zeros(self.action_space.shape, dtype=np.int32)
