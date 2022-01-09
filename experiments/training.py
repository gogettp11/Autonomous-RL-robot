import tensorflow as tf
import numpy as np
import gym
from random import choice
from collections import deque
from copy import deepcopy
from datetime import datetime
from agent_lib import MyModel, ReplayBuffer

#init enviroment
env = gym.make('CartPole-v0')
possible_actions = [i for i in range(env.action_space.n)]
qmodel_target = MyModel(env.action_space.n, env.observation_space.shape[0])
qmodel_training = MyModel(env.action_space.n, env.observation_space.shape[0])
buffer = ReplayBuffer(10000,tf.summary.create_file_writer(f"./experiments/logs/{datetime.now()}"))
#endinit enviroment
max_runs = 20000
epsilon_greedy = 1.00
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1, clipnorm=1.0)
batch_size = 64
discount_factor = 0.99
loss_function = tf.keras.losses.Huber()
update_target = 500
update_training = 3

frames = 0
loss = 0

for i in range(max_runs):
    obs = env.reset()
    done = False
    episode_reward = 0
    while not done:
        frames += 1
        t_obs = tf.expand_dims(tf.Variable(obs),axis=0)

        if epsilon_greedy < np.random.uniform():
          action = tf.argmax(tf.squeeze(qmodel_training(t_obs)))
          action = action.numpy()
        else:
          action = choice(possible_actions)

        obs , reward, done, _= env.step(action)
        buffer.add_sars(tf.squeeze(t_obs),action, reward ,tf.Variable(obs), done)
        episode_reward += reward

        #region training
        if not(frames % update_training) and buffer.size() > batch_size:
              epsilon_greedy = max(epsilon_greedy-0.001, 0.15)
              # train qmodel_target
              indices = np.random.choice(range(buffer.size()), size=batch_size)

              state_samples = tf.stack([buffer.state_history[i] for i in indices])
              state_next_samples = tf.stack([buffer.state_next_history[i] for i in indices])
              rewards_samples = tf.Variable([buffer.rewards_history[i] for i in indices])
              action_samples = [buffer.action_history[i] for i in indices]
              done_samples = tf.convert_to_tensor(
                  [float(buffer.done_history[i]) for i in indices]
              )
              future_rewards = qmodel_target(state_next_samples)
              updated_q_values = rewards_samples + discount_factor * tf.reduce_max( #bellman equation
                        future_rewards, axis=1
                    )
              updated_q_values = updated_q_values * (1 - done_samples) - done_samples #removing q values of end of episodes 

              masks = tf.one_hot(action_samples, len(possible_actions))

              with tf.GradientTape() as tape:
                        # Train the model on the states and updated Q-values
                        q_values = qmodel_training(state_samples)
                        # Apply the masks to the Q-values to get the Q-value for action taken
                        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1) # reducing q values of 2 actions to one used q value
                        # Calculate loss between new Q-value and old Q-value
                        loss = loss_function(updated_q_values, q_action)

              # Backpropagation
              grads = tape.gradient(loss, qmodel_training.trainable_variables)
              optimizer.apply_gradients(zip(grads, qmodel_training.trainable_variables))
              buffer.add_last_loss(loss, frames)
              if not(frames % update_target*update_training):
                print(f"loss: {loss} epsilon: {epsilon_greedy} model saved!")
                qmodel_target.set_weights(qmodel_training.get_weights())
                qmodel_target.save("./experiments/model")
              # if buffer.last_losses_mean() < 0.1:
              #   print(f"loss: {loss} epsilon: {epsilon_greedy} model saved!")
              #   qmodel_target.set_weights(qmodel_training.get_weights())
              #   qmodel_target.save("./experiments/model")
        #endregion
    
    #tracking all rewards
    buffer.add_episode_reward(episode_reward, i)
    mean_rewards = buffer.last_runs_mean_reward()
    if not i%12:
      print(f"mean reward after run {i}: {mean_rewards}, last loss: {loss}, epsilon: {epsilon_greedy}")
    if mean_rewards > 190:
        break