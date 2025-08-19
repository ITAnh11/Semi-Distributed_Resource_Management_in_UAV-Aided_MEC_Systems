import numpy as np
import matplotlib.pyplot as plt
from uav_mec_env import UAVMECEnv
from parameters import *
from marl.marl import MARL
from mafrl.mafrl import MAFRL
from marl.marl_ddqn import MARL_DDQN

# === Thông số mô phỏng ===
num_steps = 500  # số time slot mô phỏng mỗi case (hoặc 1000 tuỳ máy)
ue_list = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]  # số UE giống paper
# ue_list = [20, 40, 60, 80, 100]  # số UE giống paper

SEED = 42  # seed để tái lập kết quả

# Các list lưu average power với từng số lượng UE
greedy_results = []
random_results = []
local_results = []
marl_results = []
mafrl_results = []
marl_ddqn_results = []

for num_ue in ue_list:  # số UAV cố định
    print(f"Đang chạy với {num_ue} UE...")

    # Khởi tạo env với số UE cụ thể
    env = UAVMECEnv(num_ues=num_ue)

    greedy_power, random_power, local_power = [], [], []

    print("Bắt đầu MARL...")
    marl = MARL(num_ues=num_ue, num_uavs=10, num_episodes=25000)
    marl.env.reset(seed=SEED)
    marl.load_model("marl/model/marl_model_ver_ue_100.pth")
    average_energy_marl = marl.test(num_steps=num_steps)

    print("Bắt đầu MARL-DDQN...")
    marl_ddqn = MARL_DDQN(num_ues=num_ue, num_uavs=10, num_episodes=25000)
    marl_ddqn.env.reset(seed=SEED)
    marl_ddqn.load_model("marl/model/marl_ddqn_model_ver_ue.pth")
    average_energy_marl_ddqn = marl_ddqn.test(num_steps=num_steps)

    # Lưu trung bình power từng baseline
    marl_results.append(average_energy_marl)
    marl_ddqn_results.append(average_energy_marl_ddqn)

# === Vẽ biểu đồ như Fig.3 ===
plt.figure(figsize=(10, 6))
plt.plot(ue_list, marl_results, marker="o", label="MARL", linewidth=2)
plt.plot(ue_list, marl_ddqn_results, marker="*", label="MARL-DDQN", linewidth=2)

plt.xlabel("Number of UEs", fontsize=13)
plt.ylabel("Sum Power Consumption (W)", fontsize=13)
plt.title("Impact of Number of UEs on Sum Power Consumption", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig("all_vs_num_ues_marl_test.png")
