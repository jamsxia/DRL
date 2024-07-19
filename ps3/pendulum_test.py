import gymnasium as gym
from ppo import test


testing_env = gym.make("Pendulum-v1", g=9.81, render_mode="human")
test(testing_env, "model.pt")
