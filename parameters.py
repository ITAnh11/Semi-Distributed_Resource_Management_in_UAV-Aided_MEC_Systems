# parameters.py

# === System size ===
NUM_UES_K = 100
NUM_UAVS_N = 10
AREA_SIZE = 2000  # m
TIME_SLOT_DURATION = 1.0  # s
EPISODE_LENGTH = 100000  # số time slots
# === UE mobility ===
UE_SPEED_MIN = 0.0  # m/s
UE_SPEED_MAX = 1.0  # m/s
UE_MOVE_ANGLE_MIN = 0.0
UE_MOVE_ANGLE_MAX = 2 * 3.14159265359  # rad

# === UAV movement (circular pattern) ===
UAV_RADIUS = 300.0  # m
UAV_HEIGHT = 50.0  # m
UAV_SPEED = 4.0  # m/s

# === Communication parameters ===
BANDWIDTH = 1e6  # Hz
NOISE_POWER_DBM = -100  # dBm
NOISE_POWER = 1e-13  # W
PATH_LOSS_D0 = 100  # path loss reference value (dB)
MAX_TX_POWER_DBM = 20  # dBm
MAX_TX_POWER_W = 0.1  # 20 dBm = 0.1 W
NUM_TX_POWER_LEVELS = 5
TX_POWER_LEVELS = [0.02, 0.04, 0.06, 0.08, 0.1]  # W

# === UAV propulsion power ===
PROPULSION_POWER_UAV = 177  # W
BLADE_PROFILE_POWER_Pb = 80  # W
INDUCED_POWER_Pi = 90  # W
TIP_SPEED_Ut = 12  # m/s
INDUCED_SPEED_V0 = 4  # m/s
FUSELAGE_DRAG_RATIO_F0 = 0.6
ROTOR_SOLIDITY_s = 0.05
AIR_DENSITY_RHO = 1.225  # kg/m3
ROTOR_DISC_AREA_A = 0.503  # m2

# === Task parameters ===
TASK_DATA_MIN = 200e3  # bits
TASK_DATA_MAX = 800e3  # bits
TASK_CPU_MIN = 20e6  # cycles
TASK_CPU_MAX = 80e6  # cycles

# === Computation parameters ===
FMAX_UAV = 2e9  # cycles/s
CMAX_UE_PER_UAV = 30
S_J = 1e-20
OMEGA_J = 2

FMIN_UE = 1e7
FMAX_UE = 1e8
KAPPA = 2.5e-26  # W/cycle
NU = 3

# === Constraints ===
MAX_LATENCY_CONSTRAINT_TC = 1.0  # s

# === Reward weights ===
ZETA = 10.0  # UE power weight
ETA = 1.0  # UAV power weight
REWARD_PENALTY = 100  # penalty per violation

# === RL training parameters ===
NUM_HIDDEN_LAYERS = 3
LEARNING_RATE_ALPHA = 0.01
DISCOUNT_FACTOR_GAMMA = 0.7
REPLAY_MEMORY_SIZE_M = 2000
BATCH_SIZE_NB = 256
GREEDY_PROBABILITY_EPSILON = 0.9
EXPLORATION_INCREMENT_DELTA = 1e-4
GAUSSIAN_NOISE_STD = 1e-3
