# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 19:11:02 2017

@author: dell
"""
from draw import plot_reward,plot_loss
#import random
import gym
import numpy as np
from gym.wrappers import Monitor
#from collections import deque
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.optimizers import sgd
#from keras import backend as K
#EPISODES=1000
def reward_MountainCar_v0(state,next_state):
     reward = abs(next_state[0] - (-0.5))
     reward += abs(state[0] - next_state[0])
     return reward
def reward_cartpole_v1(observation_space,next_state):    
    r1 = (observation_space.high[0] - abs(next_state[0])) / observation_space.high[0] - 0.8
    r2 = (observation_space.high[2] - abs(next_state[2])) / observation_space.high[2] - 0.5
    return r1 + r2
def reward_acrobot(observation_space):  
    return 0
def  train(agent,env,EPISODES,T_num,weight_file_name,loss_file_name,reward_file_name,env_name):
    done = False
    batch_size = 32
    sum_reward=np.zeros((EPISODES))
#    history=np.zeros((EPISODES))
    loss_list=[]
    loss_EPISODES=[]
    for i_episode in range(EPISODES):
        state = env.reset()
#        state_cpy=state.copy()
        start=len(loss_list)
        state = np.reshape(state, [1, agent.state_size])
        for t in range(T_num):
            # env.render()
            action = agent.act(state)#策略选择
            next_state, reward, done, _ = env.step(action)
#            print("next_state:",state)
            if env_name=='MountainCar-v0':
                reward=reward_MountainCar_v0(state[0],next_state)
            elif env_name=='CartPole-v0':
#                print('CartPole-v0')
                reward=reward_cartpole_v1(env.observation_space,next_state)
            elif env_name=='Acrobot-v1':
                reward=reward
            else:
                reward=reward
#            r1 = (env.observation_space.high[0] - abs(next_state[0])) / env.observation_space.high[0] - 0.8
#            r2 = (env.observation_space.high[2] - abs(next_state[2])) / env.observation_space.high[2] - 0.5
#            r = r1 + r2
#            reward = r
#            reward=agent.reward_policy()
#            reward = reward if not done else 
#            if done :
#                reward = -10
            next_state = np.reshape(next_state, [1, agent.state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, i_episode: {:.2}"
                      .format(i_episode, EPISODES, t, agent.epsilon))
                sum_reward[i_episode]=t
                break
        agent.epsilon=max(agent.epsilon_min, agent.epsilon*0.995**(i_episode))
        if len(agent.memory) > batch_size:
            loss_list=agent.replay(batch_size,loss_list)
        end=len(loss_list)
#            loss_list.append()
        #        end=len(loss_list)
#        loss_EPISODES.append(np.mean(loss_list[start:end]))
#        print("loss_list,loss_EPISODES,len",start,end,loss_list,loss_EPISODES)
        loss_EPISODES.append(np.mean(loss_list[start:end]))
#        print("loss:",loss_EPISODES[i_episode])
#        print("loss_list,loss_EPISODES,len",start,end,loss_list,loss_EPISODES)
        if i_episode % 10 == 0 :
             agent.save(weight_file_name)
    plot_reward(sum_reward,reward_file_name)
    
    plot_loss(loss_EPISODES,loss_file_name)
def test(agent,env,Episode_num,T_num,weight_file_name,test_reward_file_name):
#    Episode_num=50
#    T_num=20000
    sum_reward=np.zeros((Episode_num))
#    agent = DQNAgent(state_size, action_size)
    agent.load(weight_file_name)
    for i_episode in range(Episode_num):
        observation = env.reset()
        for t in range(T_num):
            observation = np.reshape(observation, [1, agent.state_size])
            action =np.argmax(agent.model.predict(observation)[0]) 
            next_observation, reward, done, info = env.step(action)
            observation=next_observation
            if done or t==T_num-1:
                print("Episode {} finished after {} timesteps".format(i_episode + 1, t + 1))
                sum_reward[i_episode]=t
                break
    plot_reward(sum_reward,test_reward_file_name)
    print(np.mean(sum_reward),np.std(sum_reward))
    return np.mean(sum_reward),np.std(sum_reward)
def monitor(agent,env,T_num,weight_file_name,monitor_mp4):
    agent.load(weight_file_name)
    observation = env.reset()
#    env.monitor.start(monitor_mp4)
    env = Monitor(env, monitor_mp4,force=True)
    observation = env.reset()
    for t in range(T_num):
        env.render()
        observation = np.reshape(observation, [1, agent.state_size])
        action =np.argmax(agent.model.predict(observation)[0]) 
        next_observation, reward, done, info = env.step(action)
        observation=next_observation
        if done or t==T_num-1:
#            print("Episode {} finished after {} timesteps".format(i_episode + 1, t + 1))
            break
    env.close
    return 0
def test_early(agent,env,Episode_num,T_num,weight_file_name):
#    Episode_num=50
#    T_num=20000
    sum_reward=np.zeros((Episode_num))
#    agent = DQNAgent(state_size, action_size)
    agent.load(weight_file_name)
    for i_episode in range(Episode_num):
        observation = env.reset()
        for t in range(T_num):
            observation = np.reshape(observation, [1, agent.state_size])
            action =np.argmax(agent.model.predict(observation)[0]) 
            next_observation, reward, done, info = env.step(action)
            observation=next_observation
            if done or t==T_num-1:
#                print("Episode {} finished after {} timesteps".format(i_episode + 1, t + 1))
                sum_reward[i_episode]=t
                break
#    plot_reward(sum_reward,test_reward_file_name)
    print(np.mean(sum_reward),np.std(sum_reward))
    return np.mean(sum_reward),np.std(sum_reward)
def  train_early(agent,env,EPISODES,T_num,weight_file_name,loss_file_name,reward_file_name,env_name):
    done = False
    batch_size = 32
    sum_reward=np.zeros((EPISODES))
#    history=np.zeros((EPISODES))
    loss_list=[]
    loss_EPISODES=[]
    for i_episode in range(EPISODES):
        state = env.reset()
#        state_cpy=state.copy()
        start=len(loss_list)
        state = np.reshape(state, [1, agent.state_size])
        for t in range(T_num):
            # env.render()
            action = agent.act(state)#策略选择
            next_state, reward, done, _ = env.step(action)
#            print("next_state:",state)
            if env_name=='MountainCar-v0':
                reward=reward_MountainCar_v0(state[0],next_state)
            elif env_name=='CartPole-v0':
#                print('CartPole-v0')
                reward=reward_cartpole_v1(env.observation_space,next_state)
            elif env_name=='Acrobot-v1':
                reward=reward
            else:
                reward=reward
#            r1 = (env.observation_space.high[0] - abs(next_state[0])) / env.observation_space.high[0] - 0.8
#            r2 = (env.observation_space.high[2] - abs(next_state[2])) / env.observation_space.high[2] - 0.5
#            r = r1 + r2
#            reward = r
#            reward=agent.reward_policy()
#            reward = reward if not done else 
#            if done :
#                reward = -10
            next_state = np.reshape(next_state, [1, agent.state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, i_episode: {:.2}"
                      .format(i_episode, EPISODES, t, agent.epsilon))
                sum_reward[i_episode]=t
                break
        agent.epsilon=max(agent.epsilon_min, agent.epsilon*0.995**(i_episode))
        if len(agent.memory) > batch_size:
            loss_list=agent.replay(batch_size,loss_list)
            
        end=len(loss_list)
#            loss_list.append()
        #        end=len(loss_list)
#        loss_EPISODES.append(np.mean(loss_list[start:end]))
#        print("loss_list,loss_EPISODES,len",start,end,loss_list,loss_EPISODES)
        loss_EPISODES.append(np.mean(loss_list[start:end]))
#        print("loss:",loss_EPISODES[i_episode])
#        print("loss_list,loss_EPISODES,len",start,end,loss_list,loss_EPISODES)
        if i_episode % 10 == 0 :
             agent.save(weight_file_name)
             mean,std=test_early(agent,env,10,T_num,weight_file_name)
             if mean<200:
                 break
    plot_reward(sum_reward,reward_file_name)
    
    plot_loss(loss_EPISODES,loss_file_name)
    