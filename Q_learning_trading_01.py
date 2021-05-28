# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 10:17:53 2021

@author: Шевченко Роман
"""

from pathlib import Path
from collections import deque, namedtuple
from time import time
from random import sample
import numpy as np
from numpy.random import random, randint, seed
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

#import tensorflow as tf
import tensorflow.compat.v1 as tf

import gym
from gym.envs.registration import register

sns.set_style('darkgrid')

#Helper functions

#to compartable v1
tf.disable_v2_behavior()

def format_time(t):
    m_, s = divmod(t, 60)
    h, m = divmod(m_, 60)
    return '{:02.0f}:{:05.2f}'.format(m, s)

def track_results(episode, episode_nav,
                  market_nav, ratio,
                  total,
                  epsilon):
    time_ma = np.mean([episode_time[-100:]])
    T = np.sum(episode_time)

    print('{:>4d} | NAV: {:>5.3f} | Market NAV: {:>5.3f} | Delta: {:4.0f} | {} | '
          'eps: {:>6.3f}'.format(episode,
                                  episode_nav,
                                  market_nav,
                                  ratio,
                                  format_time(total),
                                  epsilon))
    

#Set up Gym enviroment
register(
    id='trading-v0',
    entry_point='trading_env:TradingEnvironment',
    max_episode_steps=1000
)

#Initialize trading enviroment
#We can instantiate the environment by using the desired trading costs and ticker:
    
trading_environment = gym.make('trading-v0')
trading_environment.env.trading_cost_bps = 1e-3
trading_environment.env.time_cost_bps = 1e-4
trading_environment.env.ticker = 'HON'
trading_environment.seed(42)

ticker = trading_environment.env.ticker

# Get Environment Params

state_dim = trading_environment.observation_space.shape[0]  # number of dimensions in state
n_actions = trading_environment.action_space.n  # number of actions
max_episode_steps = trading_environment.spec.max_episode_steps  # max number of steps per episode
    
# Define hyperparameters
gamma=.99,  # discount factor
tau=100  # target network update frequency

# NN Architecture
layers=(256,) * 3  # units per layer
learning_rate=5e-5  # learning rate
l2_reg=1e-6  # L2 regularization

# Experience Replay
replay_capacity=int(1e6)
minibatch_size=5

# epsilon-greedy Policy
epsilon_start=1.0
epsilon_end=0.1
epsilon_linear_steps=5e5
epsilon_exp_decay=.9999

# We will use TensorFlow to create our Double Deep Q-Network
#tf.reset_default_graph()
tf.compat.v1.reset_default_graph()


# The create_network function generates the three dense layers that can be trained 
# and/or reused as required by the Q network and its slower-moving target network:

def create_network(s, layers, trainable, reuse, n_actions=4):
    """Generate Q and target network with same structure"""
    for layer, units in enumerate(layers):
        x = tf.layers.dense(inputs=s if layer == 0 else x,
                            units=units,
                            activation=tf.nn.relu,
                            trainable=trainable,
                            reuse=reuse,
                            name='dense_{}'.format(layer))
    return tf.squeeze(tf.layers.dense(inputs=x,
                                      units=n_actions,
                                      trainable=trainable,
                                      reuse=reuse,
                                      name='output'))

# Placeholders
# Key elements of the DDQN's computational graph include placeholder 
# variables for the state, action, and reward sequences:
    
state = tf.placeholder(dtype=tf.float32, shape=[None, state_dim])  # input to Q network
next_state = tf.placeholder(dtype=tf.float32, shape=[None, state_dim])  # input to target network
action = tf.placeholder(dtype=tf.int32, shape=[None])  # action indices (indices of Q network output)
reward = tf.placeholder(dtype=tf.float32, shape=[None])  # rewards for target computation
not_done = tf.placeholder(dtype=tf.float32, shape=[None])  # indicators for target computation

# Episode Counter
# We add a variable to keep track of episodes:

episode_count = tf.Variable(0.0, trainable=False, name='episode_count')
add_episode = episode_count.assign_add(1)

# Deep Q Networks
# We will create two DQNs to predict q values for the current and next state, 
# where we hold the weights for the second network that's fixed when predicting 
# the next state:
    
with tf.variable_scope('Q_Network'):
    # Q network applied to current observation
    q_action_values = create_network(state,
                                     layers=layers,
                                     trainable=True,
                                     reuse=False)

    # Q network applied to next_observation
    next_q_action_values = tf.stop_gradient(create_network(next_state,
                                                           layers=layers,
                                                           trainable=False,
                                                           reuse=True))

# Slow-Moving Target Network
# In addition, we will create the target network that we update every tau periods:

with tf.variable_scope('Target_Network', reuse=False):
    target_action_values = tf.stop_gradient(create_network(next_state,
                                                           layers=layers,
                                                           trainable=False,
                                                           reuse=False))

# Collect Variables and Operations
# To build TensorFlow's computational graph, we need to collect the 
# relevant variables and operations:
    
q_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Q_Network')
target_network_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Target_Network')

# update target network weights
update_target_ops = []
for i, target_variable in enumerate(target_network_variables):
    update_target_op = target_variable.assign(q_network_variables[i])
    update_target_ops.append(update_target_op)
update_target_op = tf.group(*update_target_ops, name='update_target')

# Compute Q-Learning updates
# The target, yi, and the predicted q value is computed as follows:

# Q target calculation 
targets = reward + not_done * gamma * tf.gather_nd(target_action_values, tf.stack(
                (tf.range(minibatch_size), tf.cast(tf.argmax(next_q_action_values, axis=1), tf.int32)), axis=1))


# Estimated Q values for (s,a) from experience replay
predicted_q_value = tf.gather_nd(q_action_values,
                                 tf.stack((tf.range(minibatch_size),
                                           action), axis=1))

# Compute Loss Function
# Finally, the TD loss function that's used for stochastic 
# gradient descent is the mean squared error (MSE) between the target and prediction:
losses = tf.squared_difference(targets, predicted_q_value)
loss = tf.reduce_mean(losses)
loss += tf.add_n([tf.nn.l2_loss(var) for var in q_network_variables if 'bias' not in var.name]) * l2_reg * 0.5
    
# Tensorboard summaries
# To view results in tensorboard, we need to define summaries:
summaries = tf.summary.merge([
    tf.summary.scalar('episode', episode_count),
    tf.summary.scalar('loss', loss),
    tf.summary.scalar('max_q_value', tf.reduce_max(predicted_q_value)),
    tf.summary.histogram('loss_hist', losses),
    tf.summary.histogram('q_values', predicted_q_value)])

# Set optimizer
# We'll use the AdamOptimizer:
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss,
                                                          global_step=tf.train.create_global_step())

# Initialize TensorFlow session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# ============ Run Experiment ================

# Set parameters
total_steps, max_episodes = 0, 5000
experience = deque(maxlen=replay_capacity)
episode_time, navs, market_navs,diffs, episode_eps = [], [], [], [],[]

# Initialize variables
experience = deque(maxlen=replay_capacity)
episode_time, episode_steps, episode_rewards, episode_eps = [], [], [], []

epsilon = epsilon_start
epsilon_linear_step = (epsilon_start - epsilon_end) / epsilon_linear_steps

# Train Agent
for episode in range(max_episodes):
    episode_start = time()
    episode_reward = 0
    episode_eps.append(epsilon)

    # Initial state
    this_observation = trading_environment.reset()
    for episode_step in range(max_episode_steps):

        # choose action according to epsilon-greedy policy wrt Q

        if random() < epsilon:
            src = 'eps'
            selected_action = randint(n_actions)
        else:
            src = 'q'
            q_s = sess.run(q_action_values,
                           feed_dict={state: this_observation[None]})
            selected_action = np.argmax(q_s)

        next_observation, step_reward, done, _ = trading_environment.step(selected_action)
        episode_reward += step_reward

        # add to replay buffer
        experience.append((this_observation,
                           selected_action,
                           step_reward,
                           next_observation,
                           0.0 if done else 1.0))

        # update the target weights
        if total_steps % tau == 0:
            _ = sess.run(update_target_op)

        # update weights using minibatch of (s,a,r,s') samples from experience
        if len(experience) >= minibatch_size:
            minibatch = map(np.array, zip(
                *sample(experience, minibatch_size)))
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = minibatch

            # do a train_op with required inputs
            feed_dict = {
                state: states_batch,
                action: action_batch,
                reward: reward_batch,
                next_state: next_states_batch,
                not_done: done_batch}
            _ = sess.run([train_op],
                         feed_dict=feed_dict)

        this_observation = next_observation
        total_steps += 1

        # linearly decay epsilon from epsilon_start to epsilon_end for epsilon_linear_steps
        if total_steps < epsilon_linear_steps:
            epsilon -= epsilon_linear_step
        # then exponentially decay every episode
        elif done:
            epsilon *= epsilon_exp_decay

        if done:
            # Increment episode counter
            episode_, _ = sess.run([episode_count, add_episode])
            break

    episode_time.append(time()-episode_start)
    result = trading_environment.env.sim.result()
    final = result.iloc[-1]

    nav = final.nav * (1 + final.strategy_return)
    navs.append(nav)

    market_nav = final.market_nav
    market_navs.append(market_nav)

    diff = nav - market_nav
    diffs.append(diff)
    if episode % 250 == 0:
        track_results(episode,
                      np.mean(navs[-100:]),
                      np.mean(market_navs[-100:]),
                      np.sum([s > 0 for s in diffs[-100:]]),
                      sum(episode_time),
                      epsilon)

    if len(diffs) > 25 and all([r > 0 for r in diffs[-25:]]):
        print(result.tail())
        break

trading_environment.close()

# Store Results
results = pd.DataFrame({'episode': list(range(1, episode + 2)),
                        'nav': navs,
                        'market_nav': market_navs,
                        'outperform': diffs})

fn = 'trading_agent_result.csv'
results.to_csv(fn, index=False)

# Evaluate Results
results = pd.read_csv('trading_agent_result.csv')
results.columns = ['Episode', 'Agent', 'Market', 'difference']
results = results.set_index('Episode')
results['Strategy Wins (%)'] = (results.difference > 0).rolling(100).sum()
results.info()

fig, axes = plt.subplots(ncols=2, figsize=(14,4))
(results[['Agent', 'Market']]
 .sub(1)
 .rolling(100)
 .mean()
 .plot(ax=axes[0], 
       title='Annual Returns (Moving Average)', lw=1))
results['Strategy Wins (%)'].div(100).rolling(50).mean().plot(ax=axes[1], title='Agent Outperformance (%, Moving Average)');
for ax in axes:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))
fig.tight_layout()
# fig.savefig('figures/trading_agent', dpi=300)

