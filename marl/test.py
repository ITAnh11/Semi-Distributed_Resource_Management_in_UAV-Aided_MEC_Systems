import numpy as np
import matplotlib.pyplot as plt
from uav_mec_env import UAVMECEnv
from parameters import *
from marl.marl import MARL

# === Thông số mô phỏng ===
num_steps = 100  # số time slot mô phỏng mỗi case (hoặc 1000 tuỳ máy)
# ue_list = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]  # số UE giống paper
ue_list = [20, 40, 60]  # số UE giống paper

marl_results = []

for num_ue in ue_list:  # số UAV cố định
    print(f"Đang chạy với {num_ue} UE...")

    marl = MARL(num_ues=num_ue, num_uavs=10, num_episodes=1000)
    marl.load_model(f"marl/ver_ue_model/marl_model_{num_ue}_ues.pth")
    average_energy = marl.test(num_episodes=num_steps)

    marl_results.append(average_energy)

# === Vẽ biểu đồ như Fig.3 ===
plt.figure(figsize=(10, 6))
plt.plot(ue_list, marl_results, marker="o", label="MARL", linewidth=2)

plt.xlabel("Number of UEs", fontsize=13)
plt.ylabel("Sum Power Consumption (W)", fontsize=13)
plt.title("Impact of Number of UEs on Sum Power Consumption", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig("baseline_vs_num_ues.png")
