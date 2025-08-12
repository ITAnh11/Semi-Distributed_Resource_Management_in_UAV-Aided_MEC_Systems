from mafrl.mafrl import MAFRL

mafrl = MAFRL(num_ues=100, num_uavs=10, num_episodes=30000)
mafrl.run()
mafrl.save_model("mafrl/model/mafrl_model_ver_ue.pth")
