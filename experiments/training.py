import tensorflow as tf
import numpy as np
import gym
from random import choice
from collections import deque
from copy import deepcopy
from datetime import datetime

class ReplayBuffer():
  def __init__(self, max_memory, summary_writer = None, max_reward_tracking = 10, loss_buffer = 5) -> None:
      self.max_memory = max_memory
      self.action_history = deque(maxlen=max_memory)
      self.state_history = deque(maxlen=max_memory)
      self.state_next_history = deque(maxlen=max_memory)
      self.rewards_history = deque(maxlen=max_memory)
      self.done_history = deque(maxlen=max_memory)
      self.episode_reward_history = deque(maxlen=max_memory)
      self.last_runs_rewards = deque(iterable=[0 for i in range(max_reward_tracking)],maxlen=max_reward_tracking)
      self.last_optimalization_losses = deque(iterable=[1 for i in range(loss_buffer)],maxlen=loss_buffer)
      self.summary_writer = summary_writer
      if summary_writer:
        self.summary_writer.set_as_default()
  
  def add_sars(self,state : tf.Tensor,action:int,reward:float,next_state: tf.Tensor, done:bool): #sars'
      self.state_history.append(state)
      self.action_history.append(action)
      self.rewards_history.append(reward)
      self.state_next_history.append(next_state)
      self.done_history.append(done)
  
  def add_episode_reward(self, episode_reward : float, step: int):
      if self.summary_writer:
        tf.summary.scalar('episode_reward', episode_reward, step=step)
      self.last_runs_rewards.append(episode_reward)
  
  def last_runs_mean_reward(self):
      return np.mean(self.last_runs_rewards)
  
  def add_last_loss(self, loss : float, step : int):
      if self.summary_writer:
        tf.summary.scalar('loss', loss, step=step)
      self.last_optimalization_losses.append(loss)
  
  def last_losses_mean(self):
      return np.mean(self.last_optimalization_losses)

  def size(self):
    return len(self.state_history)

class MyModel(tf.keras.Model):

  def __init__(self, actions_n, obs_n):
    super().__init__()
    self.dense1 = tf.keras.layers.Dense(obs_n, activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(6, activation=tf.nn.relu)
    self.dense3 = tf.keras.layers.Dense(6, activation=tf.nn.tanh)
    self.dense4 = tf.keras.layers.Dense(actions_n)

  def call(self, inputs):
    x = self.dense3(self.dense2(self.dense1(inputs)))
    return self.dense4(x)

#region global variables
env = gym.make('CartPole-v0')
possible_actions = [i for i in range(env.action_space.n)]
qmodel_target = MyModel(env.action_space.n, env.observation_space.shape[0])
qmodel_training = MyModel(env.action_space.n, env.observation_space.shape[0])
max_runs = 20000
buffer = ReplayBuffer(10000,tf.summary.create_file_writer(f"./experiments/logs/{datetime.now()}"))
epsilon_greedy = 1.00
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, clipnorm=1.0)
batch_size = 64
discount_factor = 0.99
loss_function = tf.keras.losses.Huber()
frames = 0
update_target = 6000
update_training = 4
#endregion

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
              if not(frames % update_target):
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