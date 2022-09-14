import sys
import math
import random
from collections import defaultdict, deque

import gym
import numpy as np
import matplotlib.pyplot as plt

from plot_utils import plot_values

'''
This code is only used as a reference
'''

### helper functions
def plot_values(ax, V):
	# reshape the state-value function
    V = np.reshape(V, (4,12))
	# plot the state-value function
    im = ax.imshow(V, cmap='cool')
    for (j,i),label in np.ndenumerate(V):
        ax.text(i, j, np.round(label,3), ha='center', va='center', fontsize=14)
    ax.tick_params(bottom='off', left='off', labelbottom='off', labelleft='off')

def plot_scores(ax, num_episodes, scores, plot_every=100):
    ax.plot(np.linspace(0,num_episodes,len(scores),endpoint=False), np.asarray(scores))
    ax.set_xlabel('Episode Number')
    ax.set_ylabel('Average Reward (Over Next %d Episodes)' % plot_every)

def epsilon_greedy(Q, state, nA, eps):
    """Selects epsilon-greedy action for supplied state.
    
    Params
    ======
        Q (dictionary): action-value function
        state (int): current state
        nA (int): number actions in the environment
        eps (float): epsilon
    """
    if random.random() > eps: # select greedy action with probability epsilon
        return np.argmax(Q[state])
    else:                     # otherwise, select an action randomly
        return random.choice(np.arange(nA))

def update_Q_sarsamax(alpha, gamma, Q, state, action, reward, next_state=None):
    """Returns updated Q-value for the most recent experience."""
    current = Q[state][action]  # estimate in Q-table (for current state, action pair)
    Qsa_next = np.max(Q[next_state]) if next_state is not None else 0  # value of next state 
    target = reward + (gamma * Qsa_next)               # construct TD target
    new_value = current + (alpha * (target - current)) # get updated value 
    return new_value

def q_learning(env, num_episodes, alpha, gamma=1.0, plot_every=100):
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(env.nA))
    nA = env.action_space.n
    tmp_scores = deque(maxlen=plot_every)     # deque for keeping track of scores
    avg_scores = deque(maxlen=num_episodes)   # average scores over every plot_every episodes
    
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        
        score = 0                                             # initialize score
        state = env.reset()                                   # start episode
        
        eps = 1.0 / i_episode                                 # set value of epsilon (chance for exploring)
        
        while True:
            action = epsilon_greedy(Q, state, nA, eps)            # epsilon-greedy action selection
            next_state, reward, done, info = env.step(action) # take action A, observe R, S'
            score += reward    
            
            Q[state][action] = update_Q_sarsamax(alpha, gamma, Q, \
                                                state, action, reward, next_state)
            
            state = next_state     # S <- S'
            if done:
                tmp_scores.append(score)    # append score
                break
        if (i_episode % plot_every == 0):
            avg_scores.append(np.mean(tmp_scores))

    # print best 100-episode performance
    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(avg_scores)) 
    return Q, avg_scores

### The optimal Q table (or the state-action value table)
V_opt = np.zeros((4,12))
V_opt[0:13][0] = -np.arange(3, 15)[::-1]
V_opt[0:13][1] = -np.arange(3, 15)[::-1] + 1
V_opt[0:13][2] = -np.arange(3, 15)[::-1] + 2
V_opt[3][0] = -13

### Define the env
env = gym.make('CliffWalking-v0')
print(env.action_space)
print(env.observation_space)

### Q learning
Q_sarsamax, avg_scores = q_learning(env, 5000, .01)

### Print the estimated optimal policy
policy_sarsamax = np.array([np.argmax(Q_sarsamax[key]) if key in Q_sarsamax else -1 for key in np.arange(48)]).reshape((4,12))
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_sarsamax)

# Plot the estimated optimal state-value function and more
fig1, [ax1, ax2] = plt.subplots(1,2)
plot_values(ax1, V_opt)
plot_values(ax2, [np.max(Q_sarsamax[key]) if key in Q_sarsamax else 0 for key in np.arange(48)])
ax1.set_title('Optimal Q-table')
ax2.set_title('Estimated Q-table')
fig2, ax3 = plt.subplots()
plot_scores(ax3, 5000, avg_scores)
ax3.set_title('Training profile')
plt.show()
