#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bandit environment
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
2021
By Thomas Moerland
"""
import numpy as np
from BanditEnvironment import BanditEnvironment

class EgreedyPolicy:

    def __init__(self, n_actions=10):
        self.n_actions = n_actions
        self.means = np.zeros(self.n_actions)
        self.counts = np.zeros(self.n_actions) 
        pass
        
    def select_action(self, epsilon):
        #p is a random number to choose weather to choose a random action or the best action
        p = np.random.rand() 
        if p > epsilon:
            best_action = np.argmax(self.means)
            return best_action
        else:
            return np.random.randint(self.n_actions)
            
    def update(self,a,r):
        self.counts[a] += 1
        self.means[a] += (r - self.means[a]) / self.counts[a]
        pass


class OIPolicy:
    def __init__(self, n_actions=10, initial_value=5.0, learning_rate=0.1):
        self.n_actions = n_actions
        self.means = np.full(n_actions, initial_value)
        self.counts = np.zeros(n_actions)  
        self.learning_rate = learning_rate  

    def select_action(self):
        max_value = np.max(self.means)
        best_action = np.where(self.means == max_value)[0]
        a = np.random.choice(best_action)
        return a
        
    def update(self, a, r):
        self.counts[a] += 1
        self.means[a] += self.learning_rate * (r - self.means[a])
        
class UCBPolicy:

    def __init__(self, n_actions=10):
        self.n_actions = n_actions
        self.mean = np.zeros(n_actions)  
        self.counts = np.zeros(n_actions)  

    def select_action(self, c, t):
        if t < self.n_actions:
            return t
        else:
            confidence_bounds = self.mean + c * np.sqrt(np.log(t+1) / (self.counts + 1e-5))
            return np.argmax(confidence_bounds)

    def update(self, a, r):
        self.counts[a] += 1
        self.mean[a] += (r - self.mean[a]) / self.counts[a]
    

def test():
    n_actions = 10
    env = BanditEnvironment(n_actions=n_actions) # Initialize environment    
    
    pi = EgreedyPolicy(n_actions=n_actions) # Initialize policy
    a = pi.select_action(epsilon=0.5) # select action
    r = env.act(a) # sample reward
    pi.update(a,r) # update policy
    print("Test e-greedy policy with action {}, received reward {}".format(a,r))
    
    pi = OIPolicy(n_actions=n_actions,initial_value=1.0) # Initialize policy
    a = pi.select_action() # select action
    r = env.act(a) # sample reward
    pi.update(a,r) # update policy
    print("Test greedy optimistic initialization policy with action {}, received reward {}".format(a,r))
    
    pi = UCBPolicy(n_actions=n_actions) # Initialize policy
    a = pi.select_action(c=1.0,t=1) # select action
    r = env.act(a) # sample reward
    pi.update(a,r) # update policy
    print("Test UCB policy with action {}, received reward {}".format(a,r))
    
if __name__ == '__main__':
    test()
