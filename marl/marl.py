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


class MARL:
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
        self.n_observations = self.env.observation_space.shape[0]

        self.policy_net = DQN(self.n_observations, self.num_ues * self.n_actions).to(
            device
        )
        self.target_net = DQN(self.n_observations, self.num_ues * self.n_actions).to(
            device
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(REPLAY_MEMORY_SIZE_M)
        self.steps_done = 0

    def select_action(self, state, eval_mode=False):
        if eval_mode:
            with torch.no_grad():
                q_values = self.policy_net(state).view(self.num_ues, self.n_actions)
                actions = q_values.argmax(dim=1).cpu().numpy()
            return actions

        # Training mode: epsilon-greedy
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
            -1.0 * self.steps_done / EPS_DECAY
        )
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                q_values = self.policy_net(state).view(self.num_ues, self.n_actions)
                actions = q_values.argmax(dim=1).cpu().numpy()
        else:
            actions = (
                torch.randint(0, self.n_actions, (self.num_ues,), device=device)
                .cpu()
                .numpy()
            )
        return actions

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

        # Compute Q(s, a)
        q_values = self.policy_net(state_batch).view(-1, self.num_ues, self.n_actions)
        action_batch = action_batch.view(-1, self.num_ues, 1)
        state_action_values = (
            torch.gather(q_values, 2, action_batch).squeeze(-1).mean(dim=1)
        )

        # Compute Q(s', a')
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_q_values = self.target_net(non_final_next_states).view(
                -1, self.num_ues, self.n_actions
            )
            max_next_q = next_q_values.max(dim=2).values.mean(dim=1)
            next_state_values[non_final_mask] = max_next_q

        expected_q_values = reward_batch + (GAMMA * next_state_values)

        # Loss and backpropagation
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def run(self, num_steps=1000):
        print("Training started...")
        obs, _ = self.env.reset()
        state = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        for episode in range(self.num_episodes):
            if episode % 1000 == 0:
                print(f"Episode {episode}/{self.num_episodes}")

            actions = self.select_action(state)
            obs_next, reward, terminated, truncated, info = self.env.step(actions)

            reward = torch.tensor([reward], device=device)

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    obs_next, dtype=torch.float32, device=device
                ).unsqueeze(0)

            # Store the transition in memory
            action_tensor = torch.tensor(
                actions, dtype=torch.long, device=device
            ).unsqueeze(0)
            self.memory.push(state, action_tensor, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            self.optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * TAU + target_net_state_dict[key] * (1 - TAU)
            self.target_net.load_state_dict(target_net_state_dict)

        print("Training completed.")
        print("Testing the model...")
        total_energy = []
        for _ in range(num_steps):
            actions = self.select_action(state, eval_mode=True)
            obs_next, reward, done, truncated, info = self.env.step(actions)
            state = torch.tensor(
                obs_next, device=device, dtype=torch.float32
            ).unsqueeze(0)

            total_energy.append(info["total_energy_consumption"])
        print("Testing completed. Average energy consumption:", np.mean(total_energy))
        return sum(total_energy) / num_steps

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)  # Tạo thư mục nếu chưa có
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"Model loaded from {path}")

    def test(self, num_episodes=10):
        total_energy = []
        obs, _ = self.env.reset()
        state = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        for _ in range(num_episodes):
            actions = self.select_action(state, eval_mode=True)
            obs_next, reward, done, truncated, info = self.env.step(actions)
            state = torch.tensor(
                obs_next, device=device, dtype=torch.float32
            ).unsqueeze(0)

            total_energy.append(info["total_energy_consumption"])
        return sum(total_energy) / num_episodes
