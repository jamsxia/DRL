import gymnasium as gym
from ppo import train
import matplotlib
import matplotlib.pyplot as plt


training_env = gym.make("Pendulum-v1", g=9.81)
mean_rewards=train(training_env, "model.pt")

plt.plot(mean_rewards)
plot.show()

