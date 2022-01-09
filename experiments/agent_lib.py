import tensorflow as tf
import numpy as np
from collections import deque

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