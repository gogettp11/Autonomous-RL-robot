import tensorflow as tf
import numpy as np
import gym

env = gym.make('CartPole-v0')
qmodel_target = tf.keras.models.load_model('./experiments/model')
runs = 4
mean_rewards = []
#endregion

for i in range(runs):
    obs = env.reset()
    done = False
    temp_rewards = 0
    while not done:
        t_obs = tf.expand_dims(tf.Variable(obs),axis=0)
        obs , reward, done, _= env.step(tf.argmax(tf.squeeze(qmodel_target(t_obs)[0])).numpy())
        temp_rewards += reward
        env.render()
    mean_rewards.append(temp_rewards)

print(f"mean reward in {runs} runs: {np.mean(mean_rewards)}")