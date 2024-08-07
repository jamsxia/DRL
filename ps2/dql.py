## hyper paramters...

import numpy as np
import random
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Get number of actions from gym action space


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
        

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQL(object):

    def __init__(self, env):

        self.env = env

        n_actions = self.env.action_space.n

        # Get the number of state observations
        state, info = self.env.reset()
        n_observations = len(state)

        self.policy_net = DQN(n_observations, n_actions)
        self.target_net = DQN(n_observations, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.memory = ReplayMemory(10000)

        self.steps_done = 0
        self.episode_durations = []
        self.transition= namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

    def train(
            self, 
            num_episodes=600, 
            report=10,
            batch_size=128,
            gamma=0.99,
            eps_start=0.9,
            eps_end=0.05,
            eps_decay=1000,
            tau=0.005,
            lr=1e-4):
        
        optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)

        rewards=[]

        for i_episode in range(num_episodes):

            total_reward = 0

            # Initialize the environment and get its state
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            for t in count():

                action = self.select_action(state, eps_start, eps_end, eps_decay)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward])
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model(optimizer, batch_size, gamma)

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = (
                            policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau))
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_durations.append(t + 1)
                    break

                total_reward += reward.item()

            rewards.append(total_reward)

            if i_episode % report == 0:
                print('episode=%4d/%d reward=%6.6f' % (i_episode, num_episodes, total_reward))

        return rewards    

    def play(self, env):

        state, _ = env.reset()

        while True:

            env.render()

            with torch.no_grad():

                # Run the state through the policy network to get the values of the
                # different possible actions.  Then pick the action with the highest
                # value.
                action = np.argmax(self.policy_net(torch.from_numpy(state)).numpy())

                state, _, won, lost, _ = env.step(action)

                if won or lost:
                    break

    def select_action(self, state, eps_start, eps_end, eps_decay):

        sample = random.random()
        eps_threshold = eps_end + (eps_start - eps_end) * \
            np.exp(-1. * self.steps_done / eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]],dtype=torch.long)

        
    def optimize_model(self, optimizer, batch_size, gamma):

        if len(self.memory) < batch_size:
            return

        transitions = self.memory.sample(batch_size)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = self.transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)),  dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
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
        next_state_values = torch.zeros(batch_size)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        optimizer.step()
