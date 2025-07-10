import numpy as np
from parameters import *

# RMa-AV: Mô phỏng các khu vực nông thôn, nơi có mật độ dân cư và tòa nhà thấp, chủ yếu là đồng ruộng, rừng cây, hoặc các khu dân cư thưa thớt.
# UMa-AV: Khu vực đô thị vĩ mô (rộng lớn), với các tòa nhà cao tầng nhưng không quá dày đặc.
# UMi-AV: Khu vực đô thị vi mô (nhỏ, đông đúc), với các tòa nhà cao tầng và mật độ xây dựng dày đặc.
enum_type_uac = {
    "RMa-AV": 0,
    "UMa-AV": 1,
    "UMi-AV": 2,
}

carrier_frequency = [0.7, 2, 2]  # GHz, corresponding to RMa-AV, UMa-AV, UMi-AV

# [dB] (fc is in GHz and distance is in meters)


def convert_dBm_to_W(dBm):
    """Convert power from dBm to Watts."""
    return 10 ** ((dBm - 30) / 10)


def distance_2d(p1, p2):
    p1_xy = np.array(p1[:2])
    p2_xy = np.array(p2[:2])
    return np.linalg.norm(p1_xy - p2_xy)


def distance_3d(p1, p2, h_uav):
    # Ensure both p1 and p2 are 2D (x, y)
    p1_xy = np.array(p1[:2])
    p2_xy = np.array(p2[:2])
    d_2d = np.linalg.norm(p1_xy - p2_xy)
    d_3d = np.sqrt(d_2d**2 + h_uav**2)
    return d_3d


def RMa_LoS_proba(d_2d, h_uav):
    if 40 <= h_uav and h_uav <= 300:
        return 1
    p1 = max(15021 * np.log10(h_uav) - 16053, 1000)
    d1 = max(1350.8 * np.log10(h_uav) - 1602, 18)
    if d_2d <= d1:
        return 1
    P_LoS = d1 / d_2d + np.exp(-d_2d / p1) * (1 - d1 / d_2d)
    return P_LoS


def UMa_LoS_proba(d_2d, h_uav):
    if 100 <= h_uav and h_uav <= 300:
        return 1
    p1 = 4300 * np.log10(h_uav) - 3800
    d1 = max(460 * np.log10(h_uav) - 700, 18)
    if d_2d <= d1:
        return 1
    P_LoS = d1 / d_2d + np.exp(-d_2d / p1) * (1 - d1 / d_2d)
    return P_LoS


def UMi_LoS_proba(d_2d, h_uav):
    p1 = 233.98 * np.log10(h_uav) - 0.95
    d1 = max(294.05 * np.log10(h_uav) - 432.94, 18)
    if d_2d <= d1:
        return 1
    P_LoS = d1 / d_2d + np.exp(-d_2d / p1) * (1 - d1 / d_2d)
    return P_LoS


def RMa_PL_LoS(d_3d, h_uav):
    PL_LoS = max(23.9 - 1.8 * np.log10(h_uav), 20) * np.log10(d_3d) + 20 * np.log10(
        40 * np.pi * carrier_frequency[0] / 3
    )
    return PL_LoS


def UMa_PL_LoS(d_3d, h_uav):
    PL_LoS = 28.0 + 22 * np.log10(d_3d) + 20 * np.log10(carrier_frequency[1])
    return PL_LoS


def UMi_PL_LoS(d_3d, h_uav):
    PL_LoS = (
        30.9
        + (22.25 - 0.5 * np.log10(h_uav)) * np.log10(d_3d)
        + 20 * np.log10(carrier_frequency[2])
    )
    return PL_LoS


def RMa_PL_NLoS(d_3d, h_uav, PL_LoS):
    PL_NLoS = max(
        PL_LoS,
        -12
        + (35 - 5.3 * np.log10(h_uav)) * np.log10(d_3d)
        + 20 * np.log10(40 * np.pi * carrier_frequency[0] / 3),
    )
    return PL_NLoS


def UMa_PL_NLoS(d_3d, h_uav, PL_LoS):
    PL_NLoS = (
        -17.5
        + (46 - 7 * np.log10(h_uav)) * np.log10(d_3d)
        + 20 * np.log10(40 * np.pi * carrier_frequency[1] / 3)
    )
    return PL_NLoS


def UMi_PL_NLoS(d_3d, h_uav, PL_LoS):
    PL_NLoS = max(
        PL_LoS,
        32.4
        + (43.2 - 7.6 * np.log10(h_uav)) * np.log10(d_3d)
        + 20 * np.log10(carrier_frequency[2]),
    )
    return PL_NLoS


def PL(d_2d, d_3d, h_uav, env_type):
    if env_type == enum_type_uac["RMa-AV"]:
        P_LoS = RMa_LoS_proba(d_2d, h_uav)
        PL_LoS = RMa_PL_LoS(d_3d, h_uav)
        PL_NLoS = RMa_PL_NLoS(d_3d, h_uav, PL_LoS)
    elif env_type == enum_type_uac["UMa-AV"]:
        P_LoS = UMa_LoS_proba(d_2d, h_uav)
        PL_LoS = UMa_PL_LoS(d_3d, h_uav)
        PL_NLoS = UMa_PL_NLoS(d_3d, h_uav, PL_LoS)
    elif env_type == enum_type_uac["UMi-AV"]:
        P_LoS = UMi_LoS_proba(d_2d, h_uav)
        PL_LoS = UMi_PL_LoS(d_3d, h_uav)
        PL_NLoS = UMi_PL_NLoS(d_3d, h_uav, PL_LoS)

    PL = P_LoS * PL_LoS + (1 - P_LoS) * PL_NLoS
    return PL


def calculate_offloading_rate(
    ue_position, uav_position, P_TR, env_type=enum_type_uac["UMa-AV"]
):
    h_uav = uav_position[2]  # UAV height
    d_2d, d_3d = (
        distance_2d(ue_position, uav_position),
        distance_3d(ue_position, uav_position, h_uav),
    )

    PL_value = PL(d_2d, d_3d, h_uav, env_type)

    N0 = convert_dBm_to_W(
        AWGN_POWER_UAV_RECEIVER_N0
    )  # Convert noise power from dBm to W

    B = BANDWIDTH_WIRELESS_CHANNEL_B * 1e6  # Convert MHz to Hz

    rate = B * np.log2(1 + P_TR / (N0 * 10 ** (PL_value / 10)))  # bits per second
    return rate
