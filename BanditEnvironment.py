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

class BanditEnvironment:

    def __init__(self, n_actions):
        ''' Initializes a bandit environment'''
        self.n_actions = n_actions
        self.means = np.random.uniform(low=0.0,high=1.0,size=n_actions)
        #self.means = np.random.normal(loc=0.0,scale=1.0,size=n_actions) # alternative bandit definitions are possible
        self.best_action = np.argmax(self.means)
        self.best_average_return = np.max(self.means)
    
    def act(self,a):
        ''' returns a sampled reward for action a ''' 
        r = np.random.binomial(1,self.means[a])
        #r= np.random.normal(loc=self.means[a],scale=1.0) # alternative bandit definitions are possible
        return r

    
def test():
    # Initialize environment
    n_actions=10
    env = BanditEnvironment(n_actions=n_actions)
    print("Mean pay-off per action: {}".format(env.means))
    print("Best action = {} with mean pay-off {}".format(env.best_action,env.best_average_return))
    
    # Test sampling
    print('------------------------------')
    for a in range(n_actions):
        r = env.act(a)
        print('Sampled action = {}, obtained reward {}'.format(a,r))
    
if __name__ == '__main__':
    test()
