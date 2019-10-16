# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 14:21:20 2017

@author: dell
"""
import numpy as np
import matplotlib.pyplot as plt
def plot_reward(sum_reward,figname):
    plt.subplot(111)
    plt.plot(range(1,len(sum_reward)+1), sum_reward)
    plt.title('reward from episode')
    plt.xlabel('episode')
    plt.ylabel('sum_reward')
    plt.savefig(figname) 
    plt.show()
     
def plot_args(explorate_list,Learning_rate_list,figname):
    plt.subplot(111)
    plt.plot(range(1,len(explorate_list)+1), explorate_list,label="explorate_rate")
    plt.plot(range(1,len(Learning_rate_list)+1), Learning_rate_list,label="Learning_rate")
    legend = plt.legend(loc='upper center', shadow=False, fontsize='x-large')

# Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('#00FFCC')
    plt.title('rate from episode')
    plt.xlabel('episode')
    plt.ylabel('rate')
    plt.savefig(figname)  
    plt.show()
 
def plot_loss(loss_list,figname):
    plt.subplot(111)
    plt.plot(range(1,len(loss_list)+1), loss_list,label="loss")
    
    legend = plt.legend(loc='upper center', shadow=False, fontsize='x-large')
    legend.get_frame().set_facecolor('#00FFCC')
    plt.title('loss from episode')
    plt.xlabel('episode')
    plt.ylabel('loss')
    plt.savefig(figname)  
    plt.show()
    