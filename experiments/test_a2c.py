from numpy.random.mtrand import gamma
import tensorflow as tf
import numpy as np
import gym
from agent_lib import *
from datetime import datetime

env = gym.make('CartPole-v0')
possible_actions = [i for i in range(env.action_space.n)]
qmodel_training = A2C(env.action_space.n, env.observation_space.shape[0])
runs = 15000
buffer = ReplayBuffer(300, tf.summary.create_file_writer(f"./logs/{datetime.now().strftime('%H%M%S')}"))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005, clipnorm=1.0)
discount_factor = 0.95
frames = 0
q_values = []
#endregion

for episodes_count in range(runs):
    obs = env.reset()
    done = False
    while not done:
        t_obs = tf.expand_dims(tf.Variable(obs),axis=0)

        action_distribution, estimates = qmodel_training(t_obs)
        action_distribution = tf.squeeze(action_distribution).numpy()
        action = np.random.choice(possible_actions ,p=action_distribution)

        obs , reward, done, _= env.step(action)
        buffer.add_sars(tf.squeeze(t_obs),action, reward , tf.squeeze(estimates), done)
    
    buffer.rewards_history[-1] = -1
    buffer.state_next_history[-1] = 0
    # calculate state values in reverse order
    discounts = [j for j in range(buffer.size())]
    discount_factor_vec = tf.Variable([discount_factor for j in range(buffer.size())])
    reward_tensor = tf.expand_dims(tf.Variable(buffer.rewards_history), axis = 1)
    discounts_tensor = tf.expand_dims(tf.math.pow(discount_factor_vec, discounts), axis=1)
    values_tensor = tf.expand_dims(tf.stack(buffer.state_next_history), axis = 1)
    q_values = reward_tensor*discounts_tensor + values_tensor

    stacked_states = tf.stack(buffer.state_history)
    #calculate critic loss
    with tf.GradientTape() as tape:
        distribution, values = qmodel_training(stacked_states)
        advantages = q_values - values
        loss_critic = tf.math.reduce_mean(tf.math.square(advantages))
    grads_critic = tape.gradient(loss_critic, qmodel_training.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    #calculate actor loss
    with tf.GradientTape() as tape:
        distribution, values = qmodel_training(stacked_states)
        distribution_actions = tf.gather(distribution,buffer.action_history, axis = 1,batch_dims=1)
        loss_actor = tf.math.reduce_mean(advantages*tf.math.log(distribution_actions)) # minus log = gradient ascend
    grads_actor = tape.gradient(loss_actor, qmodel_training.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    # grads_critic = tape.gradient(loss_critic, qmodel_training.trainable_variables)
    optimizer.apply_gradients(zip(grads_actor+grads_critic, qmodel_training.trainable_variables))

    if not(episodes_count%50):
        print(f"episode: {episodes_count} loss: {loss_actor},{loss_critic} distribution: {distribution[-3:-1]} q_values: {advantages.numpy()}")
    buffer.add_last_loss(loss_actor, episodes_count, 'actor_loss')
    buffer.add_last_loss(loss_critic, episodes_count, 'critic_loss')
    buffer.add_episode_reward(np.sum(buffer.rewards_history),episodes_count)
    buffer.clear()

    if(buffer.last_runs_mean_reward() > 190):
        qmodel_training.save("./model")
        break