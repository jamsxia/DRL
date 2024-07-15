import numpy as np
import random
from IPython.display import clear_output

class QLAgent:
    def __init__(self, env):
        self.env=env
        self.q=q_table = np.zeros([env.observation_space.n, env.action_space.n])

    def train(self, neps, alpha, gamma , epsilon):
        plot=[]
        for i in range(1, neps):
            state = self.env.reset()
            epochs, penalties, reward, = 0, 0, 0
            done = False
            if(i%100==0):
                print(f"Iteration, Reward: {i, reward}")
                ##print(f"Reward: {reward}")
            while not done:
                if random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample() # Explore action space
                else:
                    action = np.argmax(self.q[state]) # Exploit learned values

                next_state, reward, done, info = self.env.step(action) 
                
                old_value = self.q[state, action]
                next_max = np.max(self.q[next_state])
                
                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                self.q[state, action] = new_value

                if reward == -10:
                    penalties += 1

                state = next_state
                epochs += 1
                
            if i % 100 == 0:
                clear_output(wait=True)
                
            plot[i]=reward    

        return plot

    def play(self):
        total_epochs, total_penalties = 0, 0
        episodes = 100
        state = self.env.reset()
        for _ in range(episodes):
            state = self.env.reset()
            epochs, penalties, reward = 0, 0, 0
            
            done = False
            
            while not done:
                action = np.argmax(self.q[state])
                state, reward, won, lost, info = self.env.step(action)

                if reward == -10:
                    penalties += 1

                epochs += 1

            total_penalties += penalties
            total_epochs += epochs

'''
    def chooseAction(self, state):
        
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q = [self.getQ(state, a) for a in self.actions]
            maxQ = max(q)
            count = q.count(maxQ)
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)
            action = self.actions[i]
            
        return action


    def learnQ(self, state, action, reward, value):
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha *
                        (reward + self.gamma * value - oldv)
'''
