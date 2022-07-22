import datetime
import tensorflow as tf
import numpy as np
from environment import RealWorldEnv
import pickle
import os
import random

REPLAY_MEMORY_SIZE = 100000
BATCH_SIZE = 64
MAX_STEPS = 40
EPISODES = 2000
SAVE_DATA_PATH = 'replay_memory.pkl'
SAVE_MODEL_PATH = 'q_network.h5'

# q network class
class QNetwork:
    # create the network
    def __init__(self, state_size=11, action_size=3, learning_rate=0.001, discount_factor=0.98):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.loss_function = tf.keras.losses.Huber()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

        if not self.load_model(SAVE_MODEL_PATH):
            # create the model
            self.model = tf.keras.models.Sequential()
            self.model.add(tf.keras.layers.Dense(36, input_shape=(self.state_size,), activation='tanh'))
            self.model.add(tf.keras.layers.Dense(24, activation='tanh'))
            self.model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
            self.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate)) # to create weights only
            self.save_model(SAVE_MODEL_PATH)

    # predict the action
    def predict(self, state):
        return tf.argmax(self.model(state), axis=1)
    
    # train the network
    def train(self, states, actions, rewards, next_states, done, target_model):
        # get the target values
        states = np.stack(states)
        next_states = np.stack(next_states)

        target_next = target_model(next_states)

        masks = tf.one_hot(actions, self.action_size)

        updated_q_values = rewards + self.discount_factor * tf.reduce_max(target_next, axis=1) * (1 - done)

        with tf.GradientTape() as tape:
            # Train the model on the states and updated Q-values
            q_values = self.model(states)
            # Apply the masks to the Q-values to get the Q-value for action taken
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1) # reducing q values of 3 actions to one used q value
            # Calculate loss between new Q-value and old Q-value
            loss = self.loss_function(updated_q_values, q_action)
        
        # Backpropagation
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss
    
    # save the model
    def save_model(self, path=SAVE_MODEL_PATH):
        self.model.save(path)
        print('Model saved')
    
    # load the model
    def load_model(self, path):
        # if path exists, load the model
        if os.path.exists(path):
            self.model = tf.keras.models.load_model(path)
            print('Model loaded')
            return True
        return False
    
# replay memory class
class ReplayMemory:
    def __init__(self, max_size):
        # if data exists, read it
        if not self.read():
            self.memory = []
            self.size = 0
        self.max_size = max_size
    
    def add(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])
        self.size += 1
        if self.size > self.max_size:
            self.memory.pop(0)
            self.size -= 1

    def sample(self, batch_size):
        return np.array(random.choices(self.memory, k=batch_size), dtype=object)
    
    def __len__(self):
        return self.size
    
    # read from memory
    def read(self, path=SAVE_DATA_PATH):
        # if path exists, load data
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.memory = pickle.load(f)
            self.size = len(self.memory)
            print(f'read {self.size} samples from {path}')
            return True
        return False
    
    # write to memory
    def write(self, path=SAVE_DATA_PATH):
        with open(path, 'wb') as f:
            pickle.dump(self.memory, f)
            print(f'write {self.size} samples to {path}')
        


def main():
    # init tensorboard
    writer = tf.summary.create_file_writer(f'logs/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
    writer.set_as_default()
    tf.summary.trace_on(graph=True)

    # act in the enviroment
    env = RealWorldEnv(simulation = True)
    # create the q network
    q_network = QNetwork()
    # create the replay memory
    replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE)
    # create the target network
    target_network = QNetwork()

    random_action_chance = 0.95
    min_random_action_chance = 0.05

    best_episode_reward = -100

    # play the game
    for episode in range(EPISODES):

        episode_rewards = 0
        # reset the enviroment
        state = env.reset()
        # play the game
        for step in range(MAX_STEPS):
            
            # possible random movement
            if np.random.random() < random_action_chance:
                action = np.random.randint(0, 3)
            else:
                temp_state = np.expand_dims(state, axis=0)
                action = q_network.predict(temp_state)
            # act in the enviroment
            next_state, done, reward, info = env.step(action)

            # add the experience to the replay memory
            replay_memory.add(state, action, reward, next_state, done)
            episode_rewards += reward

            state = next_state
            random_action_chance -= 0.001
            max(min_random_action_chance, random_action_chance)

            if len(replay_memory) > BATCH_SIZE and not step%4:
                # get the next experience
                batch = replay_memory.sample(BATCH_SIZE)
                # train the q network
                loss = q_network.train(batch[:,0], batch[:,1], batch[:,2], batch[:,3], batch[:,4], target_network.model)
                tf.summary.scalar('loss', loss, episode)
                # synchronize weights by 0.05
                new_weights = q_network.model.get_weights()
                old_weights = target_network.model.get_weights()
                for i in range(len(new_weights)):
                    old_weights[i] = old_weights[i] * 0.05 + new_weights[i] * 0.95
                target_network.model.set_weights(old_weights)
                # check if the game is finished
            if done:
                break
        # write reward to tensorboard
        tf.summary.scalar('episode_reward', episode_rewards, episode)
        # save the model if the results are better
        if episode_rewards > best_episode_reward:
            best_episode_reward = episode_rewards
            target_network.save_model()
            print("model saved")

        if episode_rewards > 50:
            print(f'episode {episode} finished with {episode_rewards} reward')
            break
        # save the replay memory
        replay_memory.write(SAVE_DATA_PATH)

# test the agent function
def test():
    done = False
    episode_reward = 0
    env = RealWorldEnv(simulation = True)
    q_network = QNetwork()
    state = env.reset()
    while not done:
        temp_state = np.expand_dims(state, axis=0)
        action = q_network.predict(temp_state)
        next_state, done, reward, info = env.step(action)
        episode_reward += reward
        state = next_state
    print(episode_reward)

# real world play function with the agent
def play():
    done = False
    steps = 0
    episode_reward = 0
    env = RealWorldEnv(simulation = False)
    q_network = QNetwork()
    state = env.reset()
    while not done and steps < 70:
        temp_state = np.expand_dims(state, axis=0)
        action = q_network.predict(temp_state)
        next_state, done, reward, info = env.step(action)
        episode_reward += reward
        state = next_state
        steps += 1
    with open('real_world_reward.txt', 'a') as f:
        f.write(f'episode reward {episode_reward} in {steps} steps \n')

# print something
def print_something():
    print('hello world')
    # import subprocess
    # subprocess.call(['mkdir', 'test'])
    with open('/home/gogettp11/Autonomous-RL-robot/real_world_reward.txt', 'a') as f:
        f.write(f'test\n')

if __name__ == '__main__':
    # main()
    # test()
    print_something()