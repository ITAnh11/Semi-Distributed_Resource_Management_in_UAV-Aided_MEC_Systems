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
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
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

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
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
                    f"Episode {episode}, Total Energy: {info['total_energy_consumption']:.2f} J"
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
        return np.mean(total_energy)
