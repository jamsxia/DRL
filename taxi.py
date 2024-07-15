import gymnasium as gym
from ql import QLAgent
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3")
env.render()
env.reset() # reset environment to a new, random state
agent=QLAgent(env)

agent.play()

nep=100001
plotTable=agent.train(nep, 0.1,0.6,0.1)

plt.plot([i for i in range(nep)], plotTable)

agent.play()
