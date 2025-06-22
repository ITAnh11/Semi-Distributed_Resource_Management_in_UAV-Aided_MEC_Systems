from uav_mec_env import UAVMECEnv
import numpy as np


env = UAVMECEnv(num_ues=20)
env.reset()
done = False

while not done:
    action = env.get_action_when_local_execute()
    next_state, reward, done, truncate, info = env.step(action)

    if done or truncate:
        env.render()
        env.reset()
        break
