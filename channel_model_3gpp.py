import numpy as np
from parameters import *

# Constants for 3GPP channel model
Fc = 2  # GHz  # Carrier frequency


# UMa-AV
def caculate_offloading_rate(ue_position, uav_position, tx_power, uav_height):
    d_2d = np.linalg.norm(ue_position[:2] - uav_position[:2])
    d_3d = np.sqrt(d_2d**2 + uav_height**2)
    p1 = 4300 * np.log10(uav_height) - 3800
    d1 = max(460 * np.log10(uav_height) - 700, 18)

    P_LOS = 1 if d_2d <= d1 else (d1 / d_2d + np.exp(-d_2d / p1) * (1 - d1 / d_2d))
    PL_LOS = 28 + 22 * np.log10(d_3d) + 20 * np.log10(Fc)
    PL_NLOS = (
        -17.5
        + (46 - 7 * np.log10(uav_height)) * np.log10(d_3d)
        + 20 * np.log10(40 * np.pi * Fc / 3)
    )

    PL = P_LOS * PL_LOS + (1 - P_LOS) * PL_NLOS

    rate = BANDWIDTH * np.log2(1 + tx_power / (NOISE_POWER * 10 ** (PL / 10)))
    return rate
