from marl.marl import MARL

# ue_list = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]  # số UE giống paper
ue_list = [20, 40, 60, 80]  # số UE giống paper


for num_ue in ue_list:  # số UAV cố định
    print(f"Đang chạy với {num_ue} UE...")

    marl = MARL(num_ues=num_ue, num_uavs=10, num_episodes=10000)
    # === MARL Training ===
    print("Bắt đầu huấn luyện MARL...")
    marl.run()
    print("saving model...")
    marl.save_model(f"marl/ver_ue_model/marl_model_{num_ue}_ues.pth")
