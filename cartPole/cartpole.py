import gymnasium as gym
from dql import DQL
import matplotlib
import matplotlib.pyplot as plt


testing_env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("CartPole-v1")
model=DQL(env)
model.train()

# set up matplotlib










'''print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()
'''
