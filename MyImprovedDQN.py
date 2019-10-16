# -*- coding: utf-8 -*-
import random
import argparse
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
#from draw import plot_reward,plot_loss
#from train_test_agent import test,train
EPISODES = 1000
from train_test_agent import test,train,train_early
from gym.wrappers import Monitor
class DDQNAgent:
    def __init__(self, state_size, action_size,gamma = 0.95,epsilon = 1.0,epsilon_min = 0.01,epsilon_decay = 0.995,learning_rate = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma    # discount rate
        self.epsilon = epsilon# exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

#    def _huber_loss(self, target, prediction):
#        # sqrt(1+error^2)-1
#        error = prediction - target
#        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size,loss_list):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            history=self.model.fit(state, target, epochs=1, verbose=0)
            loss_list.append(history.history['loss'][0])
#        if self.epsilon > self.epsilon_min:
#            self.epsilon *= self.epsilon_decay
        return loss_list
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN')
    parser.add_argument('--envname', type=str,
                        help='envname', default="CartPole-v0")
    args = parser.parse_args()
    envname = args.envname
#    envname='MountainCar-v0'
    env = gym.make(envname)


    if envname=="CartPole-v0":
        Episode_num=50
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        print(state_size)
        agent = DDQNAgent(state_size, action_size,learning_rate = 0.001)
        train(agent,env,1000,20000,"./save/cartpole-cartpole-ddqn.h5","cartpole-ddqn_loss_fig.png","cartpole-ddqn_reward_fig.png",envname)
        sumreward=np.zeros(Episode_num)+2000
        test(agent,env,sumreward,50,20000,"./save/cartpole-cartpole-ddqn.h5","ddqn_test_reward_fig.png")
#        monitor(agent,env,20000,"./save/cartpole-ddqn.h5",'./mp4/cartpole-Qlearning_ddqn')
    elif envname=='MountainCar-v0':
        Episode_num=100
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        agent = DDQNAgent(state_size, action_size,learning_rate = 0.1,epsilon_min = 0.2,gamma = 0.99,epsilon_decay = 0.999)
        train_early(agent,env,2000,2000,"./save/MountainCar-v0-ddqn.h5","MountainCar-v0_ddqn_loss_fig.png","MountainCar-v0_ddqn_reward_fig.png",envname)
        sumreward=np.zeros(Episode_num)+2000
        test(agent,env,sumreward,100,2000,"./save/MountainCar-v0-ddqn.h5","MountainCar-v0_ddqn_test_reward_fig.png")
    elif envname=='Acrobot-v1':
        Episode_num=100
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        print(state_size)
        agent = DDQNAgent(state_size, action_size,learning_rate = 0.001)
        train_early(agent,env,3000,2000,"./save/Acrobot-v1-dqn.h5","Acrobot-v1_dqn_loss_fig.png","Acrobot-v1_dqn_reward_fig.png",envname)
        sumreward=np.zeros(Episode_num)+2000
        test(agent,env,sumreward,100,2000,"./save/Acrobot-v1-dqn.h5","Acrobot-v1_dqn_test_reward_fig.png")
    else:
        print("{}is not exist!".format(envname))
        
