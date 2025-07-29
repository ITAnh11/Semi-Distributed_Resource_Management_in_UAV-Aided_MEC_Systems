import numpy as np
import matplotlib.pyplot as plt
from uav_mec_env import UAVMECEnv
from parameters import *
from marl.marl import MARL

# === Thông số mô phỏng ===
num_steps = 100  # số time slot mô phỏng mỗi case (hoặc 1000 tuỳ máy)
# ue_list = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]  # số UE giống paper
ue_list = [100]  # số UE giống paper

marl_results = []
list_marl_results = []
num_episodes = 50000  # số episode huấn luyện
for num_ue in ue_list:  # số UAV cố định
    print(f"Đang chạy với {num_ue} UE...")

    marl = MARL(num_ues=num_ue, num_uavs=10, num_episodes=num_episodes)
    total_energy = marl.run(num_steps=num_steps)

    plt.figure(figsize=(10, 6))
    episodes = list(range(1, len(total_energy) + 1))
    plt.plot(episodes, total_energy, label="MARL - Total Energy")

    plt.xlabel("Episode")
    plt.ylabel("Total Energy Consumption")
    plt.title("Total Energy vs. Training Episode")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("average_energy_vs_episode.png")
