import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from uav_mec_env import UAVMECEnv
from parameters import *
from dqn import DQN, ReplayMemory, Transition
import os

# Device setup
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


class MARL_DDQN:
    def __init__(self, num_ues=100, num_uavs=10, num_episodes=1000):
        self.env = UAVMECEnv(
            num_ues=num_ues,
            num_uavs=num_uavs,
            episode_length=EPISODE_LENGTH,
        )
        self.num_ues = num_ues
        self.num_uavs = num_uavs
        self.num_episodes = num_episodes

        self.n_actions = self.env.total_actions
        self.n_observations = 2 + num_uavs * 2 + 3

        self.policy_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(REPLAY_MEMORY_SIZE_M)
        self.steps_done = 0

    def select_action(self, state, eval_mode=False):
        if eval_mode:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
            -1.0 * self.steps_done / EPS_DECAY
        )
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor(
                [[self.env.action_space.sample()]], device=device, dtype=torch.long
            )

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Q(s_t, a) theo policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # === DDQN target ===
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            # 1. Chọn action tốt nhất ở next_state theo policy_net
            next_state_actions = self.policy_net(non_final_next_states).max(1).indices
            # 2. Ước lượng Q-value của action đó bằng target_net
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states)
                .gather(1, next_state_actions.unsqueeze(1))
                .squeeze(1)
            )

        # y = r + gamma * Q_target(s', a*)
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def run(self, num_steps=1000):
        print("Training started...")
        self.env.reset()
        states = []
        for ue_id in range(self.env.num_ues):
            state = torch.tensor(
                self.env.get_state_ue(ue_id), device=device, dtype=torch.float32
            ).unsqueeze(0)
            states.append(state)

        for episode in range(self.num_episodes):
            actions = []
            for ue_id in range(self.env.num_ues):
                state = states[ue_id]
                action = self.select_action(state)
                actions.append(action)

            self.steps_done += 1

            _, reward, terminated, truncated, info = self.env.step(actions)

            next_states = []
            for ue_id in range(self.env.num_ues):
                next_state = torch.tensor(
                    self.env.get_state_ue(ue_id), device=device, dtype=torch.float32
                ).unsqueeze(0)
                next_states.append(next_state)

            reward = torch.tensor(reward, device=device, dtype=torch.float32).unsqueeze(
                0
            )
            for ue_id in range(self.env.num_ues):
                self.memory.push(
                    states[ue_id], actions[ue_id], next_states[ue_id], reward
                )
            states = next_states

            self.optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            self.target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                self.target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * TAU + self.target_net_state_dict[key] * (1 - TAU)
            self.target_net.load_state_dict(self.target_net_state_dict)
            if episode % 100 == 0:
                print(
                    f"Episode {episode}, Total Energy: {info['total_energy_consumption']:.3f} J"
                )
        print("Training completed.")

        print("Testing started...")
        total_energy = []
        for step in range(num_steps):
            actions = []
            for ue_id in range(self.env.num_ues):
                state = states[ue_id]
                action = self.select_action(state, eval_mode=True)
                actions.append(action)

            _, reward, terminated, truncated, info = self.env.step(actions)

            next_states = []
            for ue_id in range(self.env.num_ues):
                next_state = torch.tensor(
                    self.env.get_state_ue(ue_id), device=device, dtype=torch.float32
                ).unsqueeze(0)
                next_states.append(next_state)

            total_energy.append(info["total_energy_consumption"])

            states = next_states
        print(f"Average Energy Consumption: {np.mean(total_energy):.3f} J")
        return np.mean(total_energy)

    def save_model(self, model_path):
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        torch.save(self.policy_net.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path):
        if os.path.exists(model_path):
            self.policy_net.load_state_dict(torch.load(model_path, map_location=device))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"Model loaded from {model_path}")
        else:
            print(f"Model file {model_path} does not exist.")
        self.policy_net.eval()

    def test(self, num_steps=1000):
        print("Testing started...")
        states = []
        for ue_id in range(self.env.num_ues):
            state = torch.tensor(
                self.env.get_state_ue(ue_id), device=device, dtype=torch.float32
            ).unsqueeze(0)
            states.append(state)

        total_energy = []
        for step in range(num_steps):
            actions = []
            for ue_id in range(self.env.num_ues):
                state = states[ue_id]
                action = self.select_action(state, eval_mode=True)
                actions.append(action)

            _, reward, terminated, truncated, info = self.env.step(actions)

            next_states = []
            for ue_id in range(self.env.num_ues):
                next_state = torch.tensor(
                    self.env.get_state_ue(ue_id), device=device, dtype=torch.float32
                ).unsqueeze(0)
                next_states.append(next_state)

            total_energy.append(info["total_energy_consumption"])

            states = next_states

        average_energy = np.mean(total_energy)
        print(f"Average Energy Consumption: {average_energy:.3f} J")
        return average_energy
