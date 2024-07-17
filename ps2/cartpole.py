import gymnasium as gym
from dql import DQL
import matplotlib
import matplotlib.pyplot as plt

training_env = gym.make("CartPole-v1")

net = DQL(training_env)

rewards = net.train()

plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
