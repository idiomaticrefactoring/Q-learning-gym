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
class DQNAgent:
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

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
#        print("self.action_size:",self.action_size)
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

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
            target = reward
#            print("target:",target)
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
#                print("new_target:",target)
            target_f = self.model.predict(state)
#            print("target_f:",target_f)
            target_f[0][action] = target
#            agent.load("./save/cartpole-dqn.h5")
            history=self.model.fit(state, target_f, epochs=1, verbose=0)
#            print("history.history['loss']",history.history['loss'])
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
        agent = DQNAgent(state_size, action_size,learning_rate = 0.001)
        train(agent,env,1000,20000,"./save/cartpole-dqn.h5","cartpole-dqn_loss_fig.png","cartpole-dqn_reward_fig.png",envname)
        sumreward=np.zeros(Episode_num)
        test(agent,env,sumreward,50,20000,"./save/cartpole-dqn.h5","cartpole-dqn_dqn_test_reward_fig.png")
    elif envname=='MountainCar-v0':
        Episode_num=100
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        print(state_size)
        agent = DQNAgent(state_size, action_size,learning_rate = 0.001)
        train_early(agent,env,2000,2000,"./save/MountainCar-v0-dqn.h5","MountainCar-v0_dqn_loss_fig.png","MountainCar-v0_dqn_reward_fig.png",'MountainCar-v0')
        sumreward=np.zeros(Episode_num)+2000
        test(agent,env,sumreward,100,2000,"./save/MountainCar-v0-dqn.h5","MountainCar-v0_dqn_test_reward_fig.png")
    elif envname=='Acrobot-v1':
        Episode_num=100
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        print(state_size)
        agent = DQNAgent(state_size, action_size,learning_rate = 0.001)
        train_early(agent,env,3000,2000,"./save/Acrobot-v1-dqn.h5","Acrobot-v1_dqn_loss_fig.png","Acrobot-v1_dqn_reward_fig.png",'Acrobot-v1')
        sumreward=np.zeros(Episode_num)+2000
        test(agent,env,sumreward,100,2000,"./save/Acrobot-v1-dqn.h5","Acrobot-v1_dqn_test_reward_fig.png")
    else:
        print("{}is not exist!".format(envname))
        
