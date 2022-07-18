import tensorflow as tf
import numpy as np
from environment import RealWorldEnv
import pickle
import os
import random

REPLAY_MEMORY_SIZE = 100000
BATCH_SIZE = 32
MAX_STEPS = 20
EPISODES = 1000

# q network class
class QNetwork:
    # create the network
    def __init__(self, state_size=11, action_size=3, learning_rate=0.001, discount_factor=0.98):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        if not self.load_model('./model/q_network.h5'):
            # create the model
            self.model = tf.keras.models.Sequential()
            self.model.add(tf.keras.layers.Dense(24, input_shape=(self.state_size,), activation='relu'))
            self.model.add(tf.keras.layers.Dense(24, activation='relu'))
            self.model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
            self.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
            self.save_model('models/q_network.h5')

    # predict the action
    def predict(self, state):
        return tf.argmax(self.model(state), axis=1)
    
    # train the network
    def train(self, states, actions, rewards, next_states, done, target_model):
        # get the target values
        target = self.model(states)
        target_next = target_model(next_states)
        target[range(self.action_size), actions] = rewards + (1 - done) * tf.math.reduce_max(target_next, axis=1)*self.discount_factor
        # train the model
        self.model.fit(states, target, epochs=1, verbose=0)
    
    # save the model
    def save_model(self, path):
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
        self.max_size = max_size
        self.memory = []
        self.size = 0
    
    def add(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])
        self.size += 1
        if self.size > self.max_size:
            self.memory.pop(0)
            self.size -= 1

    def sample(self, batch_size):
        return np.array(random.choices(self.memory, k=batch_size), dtype=list)
    
    def __len__(self):
        return self.size
    
    # read from memory
    def read(self, path):
        # if path exists, load data
        if os.path.exists(path):
            self.memory = pickle.load(path)
            self.size = len(self.memory)
            print(f'read {self.size} samples from {path}')
            return True
        return False
    
    # write to memory
    def write(self, path):
        pickle.dump(self.memory, path)
        print(f'write {self.size} samples to {path}')
        


def main():
    # act in the enviroment
    env = RealWorldEnv(simulation = True)
    # create the q network
    q_network = QNetwork()
    # create the replay memory
    replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE)
    # create the target network
    target_network = QNetwork()

    random_action_chance = 0.95

    best_episode_reward = -100

    # play the game
    for episode in range(EPISODES):

        episode_rewards = 0
        # reset the enviroment
        state = env.reset()
        # get the first state
        state = np.expand_dims(state, axis=0)
        # get the first action
        action = q_network.predict(state)
        # play the game
        for step in range(MAX_STEPS):
            
            # possible random movement
            if np.random.random() < random_action_chance:
                action = np.random.randint(0, 3)
            else:
                action = q_network.predict(state)
            # act in the enviroment
            next_state, reward, done, info = env.step(action)
            # get the next state
            next_state = np.expand_dims(next_state, axis=0)

            # add the experience to the replay memory
            replay_memory.add(state, action, reward, next_state, done)
            episode_rewards += reward

            state = next_state
            random_action_chance -= 0.005

            if len(replay_memory) > BATCH_SIZE and not step%4:
                # get the next experience
                batch = replay_memory.sample(BATCH_SIZE)
                # train the q network
                q_network.train(batch[:,0], batch[:,1], batch[:,2], batch[:,3], batch[:,4], target_network.model)
                # synchronize weights by 0.05
                new_weights = q_network.model.get_weights()
                old_weights = target_network.model.get_weights()
                for i in range(len(new_weights)):
                    old_weights[i] = old_weights[i] * 0.05 + new_weights[i] * 0.95
                target_network.model.set_weights(old_weights)
                # check if the game is finished
                if done:
                    break

        # print the result
        print("episode: {}, step: {}, reward: {}".format(episode, step, reward))
        # save the model if the results are better
        if episode_rewards > best_episode_reward:
            best_episode_reward = episode_rewards
            target_network.save_model('q_network.h5')
            print("best episode: {}, reward: {}".format(episode, best_episode_reward))
        # save the replay memory
        replay_memory.write('replay_memory.pkl')

if __name__ == '__main__':
    main()