from marl.marl import MARL
from marl.marl_ddqn import MARL_DDQN

# marl = MARL(num_ues=20, num_uavs=10, num_episodes=30000)
# marl.run()
# marl.save_model("marl/model/marl_model_ver_ue.pth")

marl_ddqn = MARL_DDQN(num_ues=20, num_uavs=10, num_episodes=30000)
marl_ddqn.run()
marl_ddqn.save_model("marl/model/marl_ddqn_model_ver_ue_20.pth")
