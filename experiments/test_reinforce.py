from numpy.random.mtrand import gamma
import tensorflow as tf
import numpy as np
import gym
from random import choice
from collections import deque
from copy import deepcopy
from agent_lib import *
from datetime import datetime

env = gym.make('CartPole-v0')
possible_actions = [i for i in range(env.action_space.n)]
qmodel_target = MyModel(env.action_space.n, env.observation_space.shape[0])
qmodel_training = MyModel(env.action_space.n, env.observation_space.shape[0])
runs = 20000
buffer = ReplayBuffer(300, tf.summary.create_file_writer(f"./logs/{datetime.now().strftime('%H%M%S')}"))
epsilon_greedy = 1.00
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
batch_size = 64
discount_factor = 0.95
loss_function = tf.keras.losses.Huber()
frames = 0
update_target = 10000
update_training = 5
q_values = []
#endregion

for episodes_count in range(runs):
    obs = env.reset()
    done = False
    while not done:
        t_obs = tf.expand_dims(tf.Variable(obs),axis=0)

        action_distribution = tf.squeeze(qmodel_training(t_obs)).numpy()
        action = np.random.choice(possible_actions ,p=action_distribution)

        obs , reward, done, _= env.step(action)
        buffer.add_sars(tf.squeeze(t_obs),action, reward ,tf.Variable(obs), done)
    
    buffer.rewards_history[-1] = -1
    mean_reward = tf.Variable(np.mean(buffer.rewards_history), dtype=tf.float32)
    q_values = []
    # calculate Q values in reverse order
    q_values.append(tf.Variable(buffer.rewards_history[-1],dtype = tf.float32))
    for j in range(2,buffer.size()+1):
        q_values.append(tf.Variable(buffer.rewards_history[-j] + discount_factor*q_values[j-2] - mean_reward))
    
    q_values = np.flip(q_values, axis = 0)
    q_values = tf.expand_dims(tf.stack(q_values), axis = 1)
    #calculate losses
    with tf.GradientTape() as tape:
        distribution = qmodel_training(tf.stack(buffer.state_history))
        distribution_actions = tf.gather(distribution,buffer.action_history, axis = 1,batch_dims=1)
        loss = tf.math.reduce_mean(q_values*-tf.math.log(distribution_actions))
    grads = tape.gradient(loss, qmodel_training.trainable_variables)
    optimizer.apply_gradients(zip(grads, qmodel_training.trainable_variables))

    if not(episodes_count%50):
        print(f"episode: {episodes_count} mean_reward: {mean_reward.numpy()}, loss: {loss}, distribution: {distribution[-3:-1]} q_values: {q_values.numpy()}")
    buffer.add_last_loss(loss, episodes_count)
    buffer.add_episode_reward(np.sum(buffer.rewards_history),episodes_count)
    buffer.clear()

    if(buffer.last_runs_mean_reward() > 190):
        qmodel_training.save("./model")
        break