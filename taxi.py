import gymnasium as gym
from ql import QLAgent
import matplotlib.pyplot as plt
import numpy as np

testing_env = gym.make("Taxi-v3", render_mode="human")

agent=QLAgent(testing_env)

#agent.play(testing_env, max_steps=20)

nep=100000

training_env = gym.make("Taxi-v3")

plotTable=agent.train(training_env, nep, 0.1, 0.6,  0.1)

agent.play(testing_env)
plt.plot(np.array([i for i in range(1,nep)]), np.array(plotTable))
plt.show()
