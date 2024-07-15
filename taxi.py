import gymnasium as gym
from ql import QLAgent
import matplotlib.pyplot as plt

testing_env = gym.make("Taxi-v3", render_mode="human")

agent=QLAgent(testing_env)

# agent.play(testing_env, max_steps=20)

nep=100000

training_env = gym.make("Taxi-v3")

agent.train(training_env, nep, 0.1, 0.6,  0.1)

agent.play(testing_env)
