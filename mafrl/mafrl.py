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


class UE_Agent:
    def __init__(self, ue_id, n_observations, n_actions):
        self.ue_id = ue_id
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.policy_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(REPLAY_MEMORY_SIZE_M)
        self.steps_done = 0

    def select_action(self, state, env, eval_mode=False):
        if eval_mode:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)

        # Training mode: epsilon-greedy
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
            -1.0 * self.steps_done / EPS_DECAY
        )
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor(
                [[env.action_space.sample()]], device=device, dtype=torch.long
            )

        return action

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

    def soft_update_target(self):
        """Soft update of the target network's weights."""
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * TAU + target_net_state_dict[key] * (1 - TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    def set_parameters(self, state_dict):
        """Set the parameters of the agent's policy network."""
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_parameters(self):
        """Get the parameters of the agent's policy network."""
        return self.policy_net.state_dict()


class Server:
    def __init__(
        self,
        n_observations: int,
        n_actions: int,
    ):
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.policy_net = DQN(self.n_observations, self.n_actions).to(device)

    def reset(self):
        """Reset the server's policy network."""
        self.policy_net = DQN(self.n_observations, self.n_actions).to(device)

    def get_parameters(self):
        return self.policy_net.state_dict()

    def set_parameters(self, state_dict):
        self.policy_net.load_state_dict(state_dict)

    def federated_aggregate(self, agents_subset):
        """Aggregate parameters from a subset of agents."""
        total_params = {
            k: torch.zeros_like(v) for k, v in self.policy_net.state_dict().items()
        }
        for agent in agents_subset:
            agent_params = agent.get_parameters()
            for k in total_params.keys():
                total_params[k] += agent_params[k]

        # Average the parameters (only over the selected agents)
        for k in total_params.keys():
            total_params[k] /= len(agents_subset)

        self.set_parameters(total_params)


class MAFRL:
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
        # self.n_observations = 3

        self.ue_agents = [
            UE_Agent(ue_id, self.n_observations, self.n_actions)
            for ue_id in range(self.num_ues)
        ]
        self.server = Server(self.n_observations, self.n_actions)

        self.aggregation_interval = 300
        self.target_update_interval = 1

    def train_and_update(self, ue_agent):
        """Train the UE agent and update its target network."""
        ue_agent.optimize_model()
        ue_agent.soft_update_target()

    def run(self):
        self.env.reset()
        for ue_agent in self.ue_agents:
            ue_agent.set_parameters(self.server.get_parameters())
        states = []
        for ue_agent in self.ue_agents:
            state = self.env.get_state_ue(ue_agent.ue_id)
            state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
            states.append(state)

        for episode in range(self.num_episodes):
            actions = []
            for ue_agent, state in zip(self.ue_agents, states):
                action = ue_agent.select_action(state, self.env)
                actions.append(action)

            _, reward, terminateds, _, infos = self.env.step(actions)
            if episode % 10 == 0:
                print(
                    f"Episode {episode + 1}/{self.num_episodes} - Total Energy: {infos['total_energy_consumption']:.3f}"
                )

            next_states = []
            reward = torch.tensor(reward, device=device, dtype=torch.float32).unsqueeze(
                0
            )
            for ue_agent in self.ue_agents:
                next_state = torch.tensor(
                    self.env.get_state_ue(ue_agent.ue_id),
                    device=device,
                    dtype=torch.float32,
                ).unsqueeze(0)
                next_states.append(next_state)

            for ue_agent in self.ue_agents:
                ue_agent.memory.push(
                    states[ue_agent.ue_id],
                    actions[ue_agent.ue_id],
                    next_states[ue_agent.ue_id],
                    reward,
                )
            states = next_states

            if episode % self.target_update_interval == 0:
                for ue_agent in self.ue_agents:
                    ue_agent.optimize_model()
                    ue_agent.soft_update_target()

            # Partial Aggregation
            if (episode + 1) % self.aggregation_interval == 0:

                self.server.federated_aggregate(self.ue_agents)
                for ue_agent in self.ue_agents:
                    ue_agent.set_parameters(self.server.get_parameters())

        # Final evaluation after training
        total_energy = 0
        for _ in range(1000):
            actions = []
            for ue_agent, state in zip(self.ue_agents, states):
                action = ue_agent.select_action(state, self.env, eval_mode=True)
                actions.append(action)

            _, reward, terminateds, _, infos = self.env.step(actions)
            total_energy += infos["total_energy_consumption"]
            next_states = []
            for ue_agent in self.ue_agents:
                next_state = torch.tensor(
                    self.env.get_state_ue(ue_agent.ue_id),
                    device=device,
                    dtype=torch.float32,
                ).unsqueeze(0)
                next_states.append(next_state)
            states = next_states

        average_energy = total_energy / 1000
        print(f"Average Energy Consumption: {average_energy:.3f} J")
        return average_energy

    def save_model(self, filename):
        """Save the model parameters to a file."""
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        torch.save(self.server.get_parameters(), filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        """Load the model parameters from a file."""
        if os.path.exists(filename):
            state_dict = torch.load(filename, map_location=device)
            self.server.set_parameters(state_dict)
            for ue_agent in self.ue_agents:
                ue_agent.set_parameters(state_dict)
            print(f"Model loaded from {filename}")
        else:
            print(f"Model file {filename} does not exist.")

    def test(self, num_steps=1000):
        """Test the trained model."""
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
                action = self.ue_agents[ue_id].select_action(
                    state, self.env, eval_mode=True
                )
                actions.append(action)

            _, reward, terminateds, _, infos = self.env.step(actions)

            next_states = []
            for ue_id in range(self.env.num_ues):
                next_state = torch.tensor(
                    self.env.get_state_ue(ue_id), device=device, dtype=torch.float32
                ).unsqueeze(0)
                next_states.append(next_state)

            total_energy.append(infos["total_energy_consumption"])

            states = next_states

        average_energy = np.mean(total_energy)
        print(f"Average Energy Consumption: {average_energy:.3f} J")
        return average_energy
