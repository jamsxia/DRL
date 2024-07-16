import numpy as np
import random


class QLAgent:

    def __init__(self, env):

        self.q=q_table = np.zeros([env.observation_space.n, env.action_space.n])

    def train(self, env, neps, alpha, gamma , epsilon, report):

        plot=[]

        for ep in range(1, neps):

            state,_ = env.reset()
            penalties, reward, = 0, 0
            done = False
            
            while not done:
                if random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample() # Explore action space
                else:
                    action = np.argmax(self.q[state]) # Exploit learned values

                next_state, reward, won, lost, info = env.step(action) 
                if won or lost:
                    break
                old_value = self.q[state, action]
                next_max = np.max(self.q[next_state])
                
                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                self.q[state, action] = new_value

                if reward == -10:
                    penalties += 1

                state = next_state
                
                
            if (ep % report==0):
                print("epoch: %d / %d | reward: %f" % (ep, neps, reward))

            plot.append(reward)
            '''if i % 100 == 0:
                clear_output(wait=True)
                '''
                
                   

        return plot

    def play(self, env, max_steps=None):

        state, _ = env.reset()

        steps = 0

        while True:

            steps += 1

            if max_steps is not None and steps > max_steps:
                break
            
            env.render()

            action = np.argmax(self.q[state])

            state, reward, won, lost, info = env.step(action)

            if won or lost:
                break

'''
    def train(self, state1, action1, reward, state2):
        maxQNew = max([self.getQ(state2, a)
        for a in self.actions])
        self.learnQ(state1, action1, reward, maxQNew)

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
