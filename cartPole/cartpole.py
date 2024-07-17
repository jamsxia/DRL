import gymnasium as gym
from dql import DQL
import matplotlib
import matplotlib.pyplot as plt


#testing_env = gym.make("CartPole-v1", render_mode="human")
testing_env = gym.make("CartPole-v1")
net=DQL(testing_env)
net.train()

# set up matplotlib










'''print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()
'''
