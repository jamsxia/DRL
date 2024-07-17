import gymnasium as gym
from dql import DQL
import matplotlib
import matplotlib.pyplot as plt

# Train the network
training_env = gym.make("CartPole-v1")
net = DQL(training_env)
rewards = net.train()

# Plot the rewards
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()

# Test the network
testing_env = gym.make("CartPole-v1", render_mode="human")
net.play(testing_env)


