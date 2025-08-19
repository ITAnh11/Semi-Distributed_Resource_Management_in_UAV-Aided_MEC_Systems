import numpy as np
import matplotlib.pyplot as plt
from uav_mec_env import UAVMECEnv
from parameters import *
from marl.marl import MARL
from mafrl.mafrl import MAFRL
from marl.marl_ddqn import MARL_DDQN
from mafrl.mafrl_ddqn import MAFRL_DDQN

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
marl_ddqn_results = []
mafrl_results = []
mafrl_ddqn_results = []


for num_ue in ue_list:  # số UAV cố định
    print(f"Đang chạy với {num_ue} UE...")

    # Khởi tạo env với số UE cụ thể
    env = UAVMECEnv(num_ues=num_ue)

    greedy_power, random_power, local_power = [], [], []

    # === Greedy Offload ===
    print("Bắt đầu Greedy Offload...")
    env.reset(seed=SEED)
    for _ in range(num_steps):
        actions = env.greedy_offloading()
        _, reward, terminated, _, info = env.step(actions)
        greedy_power.append(info["total_energy_consumption"])
        if terminated:
            env.reset()

    # === Random Offload ===
    print("Bắt đầu Random Offload...")
    env.reset(seed=SEED)
    for _ in range(num_steps):
        actions = env.random_offloading()
        _, reward, terminated, _, info = env.step(actions)
        random_power.append(info["total_energy_consumption"])
        if terminated:
            env.reset()

    # # === Local Execute ===
    print("Bắt đầu Local Execution...")
    env.reset(seed=SEED)
    for _ in range(num_steps):
        actions = env.local_execution()
        _, reward, terminated, _, info = env.step(actions)
        local_power.append(info["total_energy_consumption"])
        if terminated:
            env.reset()

    print("Bắt đầu MARL...")
    marl = MARL(num_ues=num_ue, num_uavs=10, num_episodes=25000)
    marl.env.reset(seed=SEED)
    marl.load_model("marl/model/marl_model_ver_ue_20.pth")
    average_energy_marl = marl.test(num_steps=num_steps)

    print("Bắt đầu MAFRL...")
    mafrl = MAFRL(num_ues=num_ue, num_uavs=10, num_episodes=25000)
    mafrl.env.reset(seed=SEED)
    mafrl.load_model("mafrl/model/mafrl_model_ver_ue_20.pth")
    average_energy_mafrl = mafrl.test(num_steps=num_steps)

    print("Bắt đầu MARL-DDQN...")
    marl_ddqn = MARL_DDQN(num_ues=num_ue, num_uavs=10, num_episodes=25000)
    marl_ddqn.env.reset(seed=SEED)
    marl_ddqn.load_model("marl/model/marl_ddqn_model_ver_ue_20.pth")
    average_energy_marl_ddqn = marl_ddqn.test(num_steps=num_steps)

    print("Bắt đầu MAFRL-DDQN...")
    mafrl_ddqn = MAFRL_DDQN(num_ues=num_ue, num_uavs=10, num_episodes=25000)
    mafrl_ddqn.env.reset(seed=SEED)
    mafrl_ddqn.load_model("mafrl/model/mafrl_ddqn_model_ver_ue_20.pth")
    average_energy_mafrl_ddqn = mafrl_ddqn.test(num_steps=num_steps)

    # Lưu trung bình power từng baseline
    greedy_results.append(np.mean(greedy_power))
    random_results.append(np.mean(random_power))
    local_results.append(np.mean(local_power))
    marl_results.append(average_energy_marl)
    mafrl_results.append(average_energy_mafrl)
    marl_ddqn_results.append(average_energy_marl_ddqn)
    mafrl_ddqn_results.append(average_energy_mafrl_ddqn)

# === Vẽ biểu đồ như Fig.3 ===
plt.figure(figsize=(10, 6))
plt.plot(ue_list, greedy_results, marker="^", label="Greedy Offload", linewidth=2)
plt.plot(ue_list, random_results, marker="d", label="Random Execute", linewidth=2)
plt.plot(ue_list, local_results, marker="s", label="Local Execute", linewidth=2)
plt.plot(ue_list, marl_results, marker="o", label="MARL", linewidth=2)
plt.plot(ue_list, mafrl_results, marker="x", label="MAFRL", linewidth=2)
plt.plot(ue_list, marl_ddqn_results, marker="*", label="MARL-DDQN", linewidth=2)
plt.plot(ue_list, mafrl_ddqn_results, marker="v", label="MAFRL-DDQN", linewidth=2)

plt.xlabel("Number of UEs", fontsize=13)
plt.ylabel("Sum Power Consumption (W)", fontsize=13)
plt.title("Impact of Number of UEs on Sum Power Consumption", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig("all_vs_num_ues.png")
