from uav_mec_env import UAVMECEnv
import numpy as np


for num_ues in range(20, 21, 20):
    env = UAVMECEnv(num_ues=num_ues)
    env.reset()
    done = False

    while not done:
        action = env.get_action_when_random_execute()
        next_state, reward, done, truncate, info = env.step(action)

        if done or truncate:
            env.render()
            env.reset()
            break
