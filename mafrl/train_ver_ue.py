from mafrl.mafrl import MAFRL
from mafrl.mafrl_ddqn import MAFRL_DDQN

# mafrl = MAFRL(num_ues=20, num_uavs=10, num_episodes=30000)
# mafrl.run()
# mafrl.save_model("mafrl/model/mafrl_model_ver_ue.pth")

mafrl_ddqn = MAFRL_DDQN(num_ues=20, num_uavs=10, num_episodes=30000)
mafrl_ddqn.run()
mafrl_ddqn.save_model("mafrl/model/mafrl_ddqn_model_ver_ue.pth")
