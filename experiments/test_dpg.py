import tensorflow as tf
import numpy as np
import gym
from scipy.stats import norm
from agent_lib import *
from datetime import datetime


#region global variables
env = gym.make('MountainCarContinuous-v0')

actor = DPG(env.action_space.shape[0], env.observation_space.shape[0])
actor_target = DPG(env.action_space.shape[0], env.observation_space.shape[0])

critic = DQN(1, env.action_space.shape[0] + env.observation_space.shape[0])
critic_target = DQN(1, env.action_space.shape[0] + env.observation_space.shape[0])

critic_target.set_weights(critic.get_weights())
actor_target.set_weights(actor.get_weights())

optimizer_actor = tf.keras.optimizers.Adam(learning_rate=0.005)
optimizer_critic = tf.keras.optimizers.Adam(learning_rate=0.005)

runs = 20000
buffer = ReplayBuffer(50000,tf.summary.create_file_writer(f"./experiments/logs/{datetime.now().strftime('%H%M%S')}"))
batch_size = 64
discount_factor = 0.95
loss_function = tf.keras.losses.MeanSquaredError()
frames = 0
target_update = 0.05
noise_scale = 0.5
min_noise = 0.1
max_steps = 200
#endregion

actor_loss = 0
critic_loss = 0
max_mean_reward = -300

for episodes_count in range(runs):
    episode_reward = 0
    obs = env.reset()
    done = False
    frames = 0
    while not done and frames < max_steps:
        frames += 1
        t_obs = tf.expand_dims(tf.Variable(obs, dtype=tf.float32),axis=0)

        noise = noise_scale*np.random.normal(loc=0,scale=1.0,size=env.action_space.shape[0])
        action = tf.squeeze(actor(t_obs))
        action = np.clip(action.numpy()+noise, -1, 1) #action space from 1 to -1

        obs , reward, done, _= env.step(action)
        buffer.add_sars(tf.squeeze(t_obs),action, reward ,tf.Variable(obs, dtype=tf.float32), done)
        episode_reward += reward

        #region training
        if buffer.size() > batch_size:

            indices = np.random.choice(range(buffer.size()), size=batch_size)
            state_samples = tf.stack([buffer.state_history[i] for i in indices])
            state_next_samples = tf.stack([buffer.state_next_history[i] for i in indices])
            rewards_samples = tf.Variable([buffer.rewards_history[i] for i in indices], dtype=tf.float32)
            action_samples = [buffer.action_history[i] for i in indices]
            done_samples = tf.convert_to_tensor(
                [float(buffer.done_history[i]) for i in indices]
            )
            #actor loss
            with tf.GradientTape() as tape2:
                action = actor(state_samples)
                q_values = critic(tf.concat([action, state_samples], axis = 1))
                actor_loss = -tf.reduce_mean(q_values)
            grads2 = tape2.gradient(actor_loss, actor.trainable_variables)
            #critic loss
            with tf.GradientTape() as tape1:
                future_actions = actor_target(state_next_samples)
                future_rewards = tf.concat([future_actions,state_next_samples], axis = 1)
                future_rewards = critic_target(future_rewards)
                updated_q_values = rewards_samples + discount_factor * future_rewards * (1 - done_samples) #bellman equation
                action = actor_target(state_samples)
                q_values = critic(tf.concat([action, state_samples], axis = 1))
                critic_loss = loss_function(updated_q_values, q_values)
            grads1 = tape1.gradient(critic_loss, critic.trainable_variables)

            optimizer_actor.apply_gradients(zip(grads2,actor.trainable_variables))
            optimizer_critic.apply_gradients(zip(grads1,critic.trainable_variables))

            updated_actor = np.array(actor_target.get_weights())*(1-target_update) + np.array(actor.get_weights())*(target_update)
            actor_target.set_weights(updated_actor)
            
            updated_critic = np.array(critic_target.get_weights())*(1-target_update) + np.array(critic.get_weights())*(target_update)
            critic_target.set_weights(updated_critic)
        #endregion
    
    noise_scale -= 0.005
    noise_scale = max(min_noise, noise_scale)

    buffer.add_last_loss(actor_loss, episodes_count, 'actor_loss')
    buffer.add_last_loss(critic_loss, episodes_count, 'critic_loss')
    buffer.add_episode_reward(episode_reward, episodes_count)

    if buffer.last_runs_mean_reward() > 5:
        print(f"Solved in {episodes_count}!")
        break

actor.save("./experiments/model")
