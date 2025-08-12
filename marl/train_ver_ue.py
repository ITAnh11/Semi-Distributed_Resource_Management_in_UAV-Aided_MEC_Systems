from marl.marl import MARL

marl = MARL(num_ues=100, num_uavs=10, num_episodes=30000)
marl.run()
marl.save_model("marl/model/marl_model_ver_ue.pth")
