import numpy as np
from parameters import *

# cho nay tim cung khong chac de chatgpt tu gen ra


def dbm_to_watt(dbm):
    return 10 ** ((dbm - 30) / 10)


def distance_2d(p1, p2):
    return np.linalg.norm(p1 - p2)


def distance_3d(p1, p2, h_uav):
    # Ensure both p1 and p2 are 2D (x, y)
    p1_xy = np.array(p1[:2])
    p2_xy = np.array(p2[:2])
    d_2d = np.linalg.norm(p1_xy - p2_xy)
    d_3d = np.sqrt(d_2d**2 + h_uav**2)
    return d_2d, d_3d


def P_LoS_3GPP(d_2d, h_uav):
    theta = np.degrees(np.arctan2(h_uav, d_2d))
    a = 12
    b = 0.135
    return 1 / (1 + a * np.exp(-b * (theta - a)))


def path_loss_3GPP(d_3d, f, eta_los=1.0, eta_nlos=20.0):
    c = 3e8  # m/s
    L_fs = 20 * np.log10(d_3d) + 20 * np.log10(f) + 20 * np.log10(4 * np.pi / c)
    PL_los = L_fs + eta_los
    PL_nlos = L_fs + eta_nlos
    return PL_los, PL_nlos


def PL_avg_3GPP(ue_pos, uav_pos, h_uav, f=FREQUENCY_CARRIER_F):
    d_2d, d_3d = distance_3d(ue_pos, uav_pos, h_uav)
    p_los = P_LoS_3GPP(d_2d, h_uav)
    PL_los, PL_nlos = path_loss_3GPP(d_3d, f)
    PL_avg = p_los * PL_los + (1 - p_los) * PL_nlos  # (5)
    return PL_avg


def calculate_offloading_rate(ue_pos, uav_pos, power_level):
    P_tr_max_W = dbm_to_watt(MAXIMAL_TRANSMISSION_POWER_UES_P_TR_MAX)
    N0_W = dbm_to_watt(AWGN_POWER_UAV_RECEIVER_N0)
    power = P_tr_max_W * ((power_level + 1) / NUMBER_TRANSMISSION_POWER_LEVELS_LP)
    PL_a = PL_avg_3GPP(ue_pos, uav_pos, UAV_HEIGHT_H)
    snr_linear = power / (N0_W * 10 ** (PL_a / 10))
    rate = BANDWIDTH_WIRELESS_CHANNEL_B * 1e6 * np.log2(1 + snr_linear)  # (4)
    return rate  # bit/s
