# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 15:36:37 2017

@author: dell
"""
import sys
import argparse
from gym.wrappers import Monitor
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
MIN_EXPLORE_RATE = 0.01
#
class Q_learning_agent:
    
    def __init__(self, gamma = 0.95,epsilon = 1.0,epsilon_min = 0.01,epsilon_decay = 0.995,learning_rate = 0.1):
        self.gamma = gamma    # discount rate
        self.epsilon = epsilon# exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
     
    def Discrete_state(self, upper_bound,lower_bound,state=[2,1,8,1]):
#        print(observation_space,observation_space.high,observation_space.low)
#        upper_bound=observation_space.high#position,car_v,angle,pole_v
#        lower_bound=observation_space.low
    #    upper_bound= [1.01, 1.01, 1.01, 1.01, 12.6, 28.3]
    #    lower_bound = [-1.01, -1.01, -1.01, -1.01, -12.6, -28.3]
        bins=[]
        state_num=1
        print(lower_bound,upper_bound)
        for i in range(len(upper_bound)):
            bins.append( np.linspace(lower_bound[i], upper_bound[i], state[i]))
            state_num*=state[i]
    #    print("state_num:",state_num)
        return bins,state_num
    def Digitize_state(self,observation,bins)   :
        state_index=0
        b=1
        for i, feature in enumerate(observation):
            if len(bins[i]>1):
                a= np.digitize(feature, bins[i])
                if a==0 or a==len(bins[i]):
                    a=0
            else:
                a=0
    #        print(feature, bins[i],a)
            if i>0:
                b=b*(len(bins[i-1]))
            state_index+=a*b
    #    print("observation,state_index:",observation,state_index)
        return state_index
    def policy_greedy(self,state_index,Explor_rate,action_num,Q_val):
        enable_explo=Explor_rate>np.random.uniform(0, 1)
        if enable_explo:        
            action=np.random.randint(0, action_num)
    #        print(action)
        else:
             action=np.argmax(Q_val[state_index])
    #        action=pi[state_index]
        return action
    def plot_reward(sum_reward):
        plt.subplot(111)
        plt.plot(range(1,len(sum_reward)+1), sum_reward)
        plt.title('reward from episode')
        plt.xlabel('sum_reward')
        plt.ylabel('episode')
        plt.show()
    def plot_args(explorate_list,Learning_rate_list):
        plt.subplot(111)
        plt.plot(range(1,len(explorate_list)+1), explorate_list,label="explorate_rate")
        plt.plot(range(1,len(Learning_rate_list)+1), Learning_rate_list,label="Learning_rate")
        legend = plt.legend(loc='upper center', shadow=False, fontsize='x-large')
    
    # Put a nicer background color on the legend.
        legend.get_frame().set_facecolor('#00FFCC')
        plt.title('rate from episode')
        plt.xlabel('episode')
        plt.ylabel('rate')
        plt.show()
    def get_explore_rate(self,MIN_EXPLORE_RATE,t,Explor_rate):
        return max(MIN_EXPLORE_RATE, Explor_rate*0.995**(t))

    def get_learning_rate(self,MIN_LEARNING_RATE,t,Learning_rate):
        return max(MIN_LEARNING_RATE, Learning_rate*0.995**(t))

def test(env,agent,Q_val,bins,Episode_num,sum_reward,T_num=20000):
#        Episode_num=50
#        sum_reward=np.zeros((Episode_num))
    for i_episode in range(Episode_num):
#        if i_episode==Episode_num-1:
#            env = Monitor(env, './mp4/cartpole-Qlearning2',force=True)
        observation = env.reset()
        state_index=agent.Digitize_state(observation,bins)   
        for t in range(T_num):
#            env.render()
            action=np.argmax(Q_val[state_index])
            observation, reward, done, info = env.step(action)
            new_state_index=agent.Digitize_state(observation,bins)
            state_index=new_state_index
            if done or t==T_num-1:
#                print("Episode {} finished after {} timesteps".format(i_episode + 1, t + 1))
                sum_reward[i_episode]=t
                break
#        plot_reward(sum_reward)
    print(np.mean(sum_reward),np.std(sum_reward))
    return 0
def  train(agent,env,Episode_num,T_num,envname,upp,low,state_num=[2,2,10,5]):
#    env = gym.make(name)
#    env = env.unwrapped # 不做这个会有很多限制
#    env.seed(0)
    action_num=env.action_space.n
    
    bins,state_num=agent.Discrete_state(upp,low,state_num)   
    np.save("bins.npy",bins)
    Q_val=np.zeros((state_num,action_num))
    
    my_Explor_rate=agent.epsilon
    my_Learning_rate=agent.learning_rate
    mean_num=200
    
    print(Episode_num/mean_num)
    sum_reward=np.zeros((Episode_num))
    explorate_list=np.zeros((Episode_num))
    Learning_rate_list=np.zeros((Episode_num))
    for i_episode in range(Episode_num):    
        explorate_list[i_episode]=my_Explor_rate
        Learning_rate_list[i_episode]=my_Learning_rate

        observation = env.reset()
        state_index=agent.Digitize_state(observation,bins)   

        for t in range(T_num):
#            env.render()
            action=agent.policy_greedy(state_index,my_Explor_rate,action_num,Q_val)
#            print(action,state_index)
            observation, reward, done, info = env.step(action)
            
            new_state_index=agent.Digitize_state(observation,bins)
#            new_action=pi[new_state_index]
            new_action=np.argmax(Q_val[new_state_index])
            
            if done and t<T_num-1 and envname=='CartPole-v0':
                reward= -20000
                
                
            Q_val[state_index][action]+=my_Learning_rate*(reward+agent.gamma*(Q_val[new_state_index][new_action])-Q_val[state_index][action])
            state_index=new_state_index
            if done:
                sum_reward[i_episode]=t
                print("Episode {} finished after {} timesteps.{}.{}".format(i_episode + 1, t + 1,agent.epsilon,agent.epsilon))
                break
        my_Explor_rate = agent.get_explore_rate(MIN_EXPLORE_RATE,i_episode,agent.epsilon)
#        my_Learning_rate=get_learning_rate(MIN_LEARNING_RATE,i_episode,Learning_rate)
#    plot_reward(sum_reward)
    np.save("q_table_2.npy",Q_val)
#    print(explorate_list)
#    plot_args(explorate_list,Learning_rate_list)
    for i in range(int(Episode_num/mean_num)):
        print(np.mean(sum_reward[i*mean_num:(i+1)*mean_num]),np.std(sum_reward[i*mean_num:(i+1)*mean_num]))
    return Q_val,bins,env
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Q-learning')
    parser.add_argument('--envname', type=str,
                        help='envname', default="CartPole-v0")
    args = parser.parse_args()
    envname = args.envname
    envname='MountainCar-v0'
    env = gym.make(envname)
    
    if envname=="CartPole-v0":
        Episode_num=50
        upper_bound=env.observation_space.high#[position of cart, velocity of cart, angle of pole, rotation rate of pole]
        lower_bound=env.observation_space.low
        upper_bound[1]=1.0
        lower_bound[1]=-1.0
        upper_bound[3]=1.0
        lower_bound[3]=-1.0
        agent = Q_learning_agent(learning_rate=0.1,epsilon=1.0 ,gamma=0.95)
        Q_val,bins=train(agent,env,2000,2000,envname,upper_bound,lower_bound,state_num=[8,8,8,8])
        sumreward=np.zeros(Episode_num)
        test(env,agent,Q_val,bins,Episode_num,sumreward,T_num=20000)
    elif envname=='MountainCar-v0':
        Episode_num=100
        upper_bound=env.observation_space.high#position,car_v,angle,pole_v
        lower_bound=env.observation_space.low
        agent = Q_learning_agent(learning_rate=0.1,epsilon=1.0 ,gamma=0.95)
        train(agent,env,10000,200,envname,upper_bound,lower_bound,state_num=[40,40])
        test(env,agent,Q_val,bins,Episode_num,sumreward,T_num=2000)
    elif envname=='Acrobot-v1':
        Episode_num=100
        upper_bound=env.observation_space.high#position,car_v,angle,pole_v
        lower_bound=env.observation_space.low
        agent = Q_learning_agent(learning_rate=0.1,epsilon=1.0 ,gamma=0.95)
        train(agent,env,3000,200,envname,upper_bound,lower_bound,state_num=[4,4,4,4,10,10])
        test(env,agent,Q_val,bins,Episode_num,sumreward,T_num=2000)
    else:
        print("{}is not exist!".format(envname))
            
#    Q_val,bins,env=exper('CartPole-v0',state_num=[8,8,8,8],Discount_factor=0.95)
#    
##    test(env,Q_val,bins,T_num=20000)
#
#    Q_val,bins,env=exper_mountain('MountainCar-v0',10000,200,Learning_rate=0.1,Explor_rate=1.0,state_num=[40,40],Discount_factor=0.95)
#    Q_val,bins,env=exper_mountain('Acrobot-v1',3000,200,Learning_rate=0.01,Explor_rate=1.0,state_num=[4,4,4,4,10,10],Discount_factor=0.97)
#==============================================================================进行监控录屏
#     env = gym.make('CartPole-v0')
# ##    env = env.unwrapped # 不做这个会有很多限制
#     Q_val=np.load("q_table_2.npy")
#     bins=np.load("bins.npy")
#     monitor(env,20000,"q_table_2.npy",'./mp4/cartpole-Qlearning')
#==============================================================================
#    test(env,Q_val,bins,T_num=20000)
#    env.monitor.start('/mp4/cartpole-Qlearning')
#    env = Monitor(env, './mp4/cartpole-Qlearning2',force=True)
#    Episode_num=1
#    T_num=20
#    sum_reward=np.zeros((Episode_num))
#    for i_episode in range(Episode_num):
#        observation = env.reset()
#        state_index=Digitize_state(observation,bins)   
#        for t in range(T_num):
##            env.render()
#            action=np.argmax(Q_val[state_index])
#            observation, reward, done, info = env.step(action)
#            new_state_index=Digitize_state(observation,bins)
#            state_index=new_state_index
#            if done:
#                print("Episode {} finished after {} timesteps.{}.{}".format(i_episode + 1, t + 1,Learning_rate,Explor_rate))
#                sum_reward[i_episode]=t
#                break
#    env.close
#    test_mountain(env,Q_val,bins,T_num=2000)
   
#    Q_val,bins,env=exper('MountainCar-v0',state_num=[8,10],Discount_factor=0.99)
 
#    test(env,Q_val,bins,T_num=2000)
#    Q_val,bins,env=exper('MountainCar-v0',state_num=[8,10],Discount_factor=0.99)
             
#    test(env,Q_val,bins,T_num=2000)
