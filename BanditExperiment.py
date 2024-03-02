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
import matplotlib.pyplot as plt
from BanditEnvironment import BanditEnvironment
from BanditPolicies import EgreedyPolicy, OIPolicy, UCBPolicy 

########################### EPSILON GREEDY EXPERIMENT ############################

# running one bandit experiment
def run_experiment_egreedy(policy, environment, epsilon, n_timesteps):
    rewards = np.zeros(n_timesteps)
    
    for t in range(n_timesteps):
        action = policy.select_action(epsilon)  
        reward = environment.act(action)
        policy.update(action, reward)
        rewards[t] = reward
    return rewards

# running repetitions of the experiment
def run_repetitions_egreedy(n_actions, epsilon, n_timesteps, n_rep):
    all_rewards = np.zeros((n_rep, n_timesteps))
    
    for i in range(n_rep):
        environment = BanditEnvironment(n_actions)
        policy = EgreedyPolicy(n_actions)  
        all_rewards[i] = run_experiment_egreedy(policy, environment, epsilon, n_timesteps) 
    return all_rewards

def smooth_rewards(all_rewards, smoothing_window):
    smoothed_rewards = np.convolve(np.mean(all_rewards, axis=0), np.ones(smoothing_window)/smoothing_window, mode='valid')
    return smoothed_rewards

def plot_results(smoothed_rewards, epsilon, y_lim=(0.0, 1.0)):
    plt.plot(smoothed_rewards, label=f'$\\epsilon={epsilon}$')
    plt.ylim(y_lim)


plt.figure(figsize=(12, 8))
epsilon_values = [0.01, 0.05, 0.1, 0.25]
for epsilon in epsilon_values:
    all_rewards = run_repetitions_egreedy(10, epsilon, 1000, 500)
    smoothed_rewards = smooth_rewards(all_rewards, 31)
    plot_results(smoothed_rewards, epsilon)

plt.title('Average Rewards Over Time for Different Epsilons')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.legend()
plt.savefig('epsilon_greedy_experiment.png')
plt.clf()

########################### OPTIMISTIC INITIALISATION EXPERIMENT ###########################

# running one bandit experiment
def run_experiment_oi(policy_class, environment, initial_value, n_timesteps):
    policy = policy_class(environment.n_actions, initial_value)
    rewards = np.zeros(n_timesteps)
    
    for t in range(n_timesteps):
        action = policy.select_action()
        reward = environment.act(action)
        policy.update(action, reward)
        rewards[t] = reward
    return rewards

# running repetitions of the experiment
def run_repetitions_oi(policy_class, n_actions, initial_values, n_timesteps, n_rep, smoothing_window):
    avg_rewards = {}
    
    for initial_value in initial_values:
        all_rewards = np.zeros((n_rep, n_timesteps))
        for i in range(n_rep):
            environment = BanditEnvironment(n_actions)
            rewards = run_experiment_oi(policy_class, environment, initial_value, n_timesteps)
            all_rewards[i] = rewards

        smoothed_rewards = np.mean(all_rewards, axis=0)
        smoothed_rewards = np.convolve(smoothed_rewards, np.ones(smoothing_window)/smoothing_window, mode='valid')
        avg_rewards[initial_value] = smoothed_rewards
    return avg_rewards

initial_values = [0.1, 0.5, 1.0, 2.0]
avg_rewards = run_repetitions_oi(OIPolicy, 10, initial_values, 1000, 500, 31)

plt.figure(figsize=(10, 6))

for initial_value, rewards in avg_rewards.items():
    plt.plot(rewards, label=f'Initial Value = {initial_value}')

plt.title('Average Rewards Over Time for Different Initial Values')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.ylim(0.0, 1.0)
plt.legend()
plt.savefig('optimistic_initialisation_experiment.png')
plt.clf()



##################################### UCB EXPERIMENT ##########################################

# running one bandit experiment
def run_experiment_ucb(policy, environment, c, n_timesteps):
    rewards = np.zeros(n_timesteps)
    
    for t in range(n_timesteps):
        action = policy.select_action(c, t)  
        reward = environment.act(action)
        policy.update(action, reward)
        rewards[t] = reward
    return rewards

# running repetitions of the experiment
def run_repetitions_ucb(n_actions, c, n_timesteps, n_rep):
    all_rewards = np.zeros((n_rep, n_timesteps))
    
    for i in range(n_rep):
        environment = BanditEnvironment(n_actions)
        policy = UCBPolicy(n_actions)
        all_rewards[i] = run_experiment_ucb(policy, environment, c, n_timesteps)
    return all_rewards

def smooth_rewards(all_rewards, smoothing_window):
    smoothed_rewards = np.convolve(np.mean(all_rewards, axis=0), np.ones(smoothing_window)/smoothing_window, mode='valid')
    return smoothed_rewards

plt.figure(figsize=(12, 8))

c_values = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
for c in c_values:
    all_rewards = run_repetitions_ucb(10, c, 1000, 500)
    smoothed_rewards = smooth_rewards(all_rewards, 31)
    plt.plot(smoothed_rewards, label=f'c={c}')

plt.title('Average Rewards Over Time for Different Values of c')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.ylim(0.0, 1.0)
plt.legend()
plt.savefig('ucb_experiment.png')
plt.clf()


####################################### COMPARISON PLOT FOR DIFFERENT PARAMETR VALUES ##############################################

def get_avg_reward(policy_class, parameter_values, n_actions, n_timesteps, n_rep, policy_type):
    avg_rewards = {}
    for value in parameter_values:
        total_reward = 0
        for _ in range(n_rep):
            # Initializing the policy based on the type
            if policy_type == 'egreedy':
                policy_class = EgreedyPolicy(n_actions=n_actions)
            elif policy_type == 'ucb':
                policy_class = UCBPolicy(n_actions=n_actions)
            elif policy_type == 'oi':
                policy_class = OIPolicy(n_actions=n_actions, initial_value=value)

            environment = BanditEnvironment(n_actions)  
            for t in range(n_timesteps):
                if policy_type in ['egreedy', 'ucb']:
                    action = policy_class.select_action(value) if policy_type == 'egreedy' else policy_class.select_action(value, t)
                elif policy_type == 'oi':
                    action = policy_class.select_action() #OI policy diesnt accept anything in select_action()
                reward = environment.act(action)
                policy_class.update(action, reward)
                total_reward += reward
        avg_rewards[value] = total_reward / (n_rep * n_timesteps) 
    return avg_rewards

#different settings for policies to compare 
epsilon_values = np.array([0.01, 0.05, 0.1, 0.25])
initial_values = np.array([0.1, 0.5, 1.0, 2.0])
c_values = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 1.0])

avg_rewards_egreedy = get_avg_reward(EgreedyPolicy, epsilon_values, 10, 1000, 500, 'egreedy')
avg_rewards_oi = get_avg_reward(OIPolicy, initial_values, 10, 1000, 500, 'oi')
avg_rewards_ucb = get_avg_reward(UCBPolicy, c_values, 10, 1000, 500, 'ucb')

plt.figure(figsize=(12, 8))
plt.plot(epsilon_values, list(avg_rewards_egreedy.values()), marker='o', linestyle='-', label='Epsilon-Greedy', color='#FFB6C1')
plt.plot(initial_values, list(avg_rewards_oi.values()), marker='s', linestyle='-', label='Optimistic Initialization', color='magenta')
plt.plot(c_values, list(avg_rewards_ucb.values()), marker='^', linestyle='-', label='UCB', color='purple')
plt.xscale('log')
plt.title('Parameter Study of Bandit Policies')
plt.xlabel('Parameter Values')
plt.ylabel('Average Reward over 1000 steps')
plt.legend()
plt.savefig('parameter_study_bandit_policies.png') 
plt.clf()

################################################ OPTIMAL LEARNING CURVE ###############################################################

def run_experiment(policy, environment, parameter, n_timesteps):
    rewards = np.zeros(n_timesteps)
    for t in range(n_timesteps):
        if isinstance(policy, EgreedyPolicy) or isinstance(policy, UCBPolicy):
            action = policy.select_action(parameter, t) if isinstance(policy, UCBPolicy) else policy.select_action(parameter)
        else:
            action = policy.select_action()
        reward = environment.act(action)
        policy.update(action, reward)
        rewards[t] = reward
    return rewards

def run_repetitions(policy_class, parameter, n_actions, n_timesteps, n_rep, policy_type):
    all_rewards = np.zeros((n_rep, n_timesteps))
    for i in range(n_rep):
        environment = BanditEnvironment(n_actions)
        # Initializing the policy based on the type
        if policy_type == 'egreedy':
            policy = EgreedyPolicy(n_actions)
        elif policy_type == 'oi':
            policy = OIPolicy(n_actions, initial_value=parameter)
        elif policy_type == 'ucb':
            policy = UCBPolicy(n_actions)
        all_rewards[i] = run_experiment(policy, environment, parameter, n_timesteps)
    return all_rewards

def smooth_rewards(all_rewards, smoothing_window):
    return np.convolve(np.mean(all_rewards, axis=0), np.ones(smoothing_window)/smoothing_window, mode='valid')

# Optimal parameters derived based on the comparison plot
epsilon_optimal = 0.05
initial_value_optimal = 0.5
c_optimal = 0.25

rewards_egreedy = run_repetitions(EgreedyPolicy,epsilon_optimal , 10, 1000, 500, 'egreedy')
rewards_oi = run_repetitions(OIPolicy, initial_value_optimal, 10, 1000, 500, 'oi')
rewards_ucb = run_repetitions(UCBPolicy, c_optimal, 10, 1000, 500, 'ucb')

smoothed_rewards_egreedy = smooth_rewards(rewards_egreedy, 31)
smoothed_rewards_oi = smooth_rewards(rewards_oi, 31)
smoothed_rewards_ucb = smooth_rewards(rewards_ucb, 31)

plt.figure(figsize=(12, 8))
plt.plot(smoothed_rewards_egreedy, label='Epsilon-Greedy with epsilon=0.1')
plt.plot(smoothed_rewards_oi, label='Optimistic Initialization with initial_value=0.5')
plt.plot(smoothed_rewards_ucb, label='Upper Confidence Bound with c=0.25')
plt.title('Optimal Learning Curves')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.ylim(0.0, 1.0) 
plt.legend()
plt.savefig('optimal_learning_curves.png')
plt.clf()
