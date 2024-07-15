import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import torch
from algorithms.ppo.ppo import Agent
from algorithms.ppo import utils
import time
def ppo_gene_test_runner(env,config):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = "cpu"
    agent = Agent(state_dim, config['hidden_dim'], action_dim, config['actor_lr'], config['critic_lr'], config['tau'], config['gamma'],config['capacity'],config['batch_size'],device)
    test_path = "./test_results/ppo/"
    utils.ensure_directory_exists(test_path)
    agent.load_model()
    rewards = []
    test_episodes = 100
    for ep in tqdm(range(test_episodes)):
        state,info = env.reset()
        done = False
        total_rewards = 0
        while not done:
            action, action_prob = agent.select_action(state)
            next_state, reward, terminated,truncted, info = env.step(action)
            done = terminated or truncted
            state = next_state
            total_rewards += reward
            if done :
                break
        rewards.append(total_rewards)
    rewards = rewards / 5 - 100
    smoothed_rewards = utils.smooth(rewards,50)
    plt.plot(smoothed_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('PPO Reward over Episodes')
    plt.savefig('./test_results/ppo/test_ppo_best_gene_reward_plot.png')  # Save the plot as an image
    plt.clf()  # Clear the plot for the next update

class ppoRunner(object):
    def __init__(self,state_dim,action_dim,hidden_dim,actor_lr,critic_lr,gamma,tau,seed,episodes,capacity,batch_size,device='cpu'):
        self.num_evaluations = 100
        self.device = device    
        self.render = False     # 训练过程中演示环境
        self.draw = False       # 训练过程中画图
        # gene
        self.hidden_dim = hidden_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.episodes = episodes
        self.capacity = capacity
        self.batch_size = batch_size
        self.seed = seed

        self.agent = Agent(state_dim, self.hidden_dim, action_dim, self.actor_lr, self.critic_lr, self.tau, self.gamma,self.capacity,self.batch_size,self.device)

    def save_model(self):
        self.agent.save_model()
        print('Save model successfully')
        #save the gene to the txt file
        with open('./configs/ppo_best_gene_config.yaml', 'w') as f:
            f.write("algo: ppo\n")
            f.write("hidden_dim: "+str(self.hidden_dim) + '\n')
            f.write("actor_lr: "+str(self.actor_lr) + '\n')
            f.write("critic_lr: "+str(self.critic_lr) + '\n')
            f.write("gamma: "+str(self.gamma) + '\n')
            f.write("tau: "+str(self.tau) + '\n')
            f.write("episodes: "+str(self.episodes) + '\n')
            f.write("capacity: "+str(self.capacity) + '\n')
            f.write("batch_size: "+str(self.batch_size) + '\n')
            f.write("seed: "+str(self.seed) + '\n')

    def train(self,env):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        train_path = "./train_results/ppo/"
        test_path = "./test_results/ppo/"
        utils.ensure_directory_exists(train_path)
        utils.ensure_directory_exists(test_path)

        rewards = []
        losses_actor = []
        losses_critic = []

        for ep in tqdm(range(self.episodes)):
            state,info = env.reset()
            done = False
            totral_reward = 0
            loss_a = 0
            loss_c = 0
            while not done:
                action, action_prob = self.agent.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                if self.render == True:
                    env.render()
                self.agent.memory(state, action, action_prob, reward, next_state)
                state = next_state
                totral_reward += reward
                if done :    
                    break
            loss_a,loss_c = self.agent.update()
            rewards.append(totral_reward)
            losses_actor.append(loss_a)
            losses_critic.append(loss_c)
                
        self.agent.save_model()
        rewards = [each/5 - 100 for each in rewards]
        smoothed_rewards = utils.smooth(rewards,50)
        plt.plot(smoothed_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Reward over Episodes')
        plt.savefig('./train_results/ppo/smoothed_reward_plot_{}.png'.format(time.time()%2000))  # Save the plot as an image
        plt.clf()  # Clear the plot for the next update

        # smoothed_actor_loss = utils.smooth(losses_actor)
        # plt.plot(smoothed_actor_loss)
        # plt.xlabel('Episode')
        # plt.ylabel('Loss')
        # plt.title('Loss over Episodes')
        # plt.savefig('./train_results/ppo/smoothed_actor_loss_plot_{}.png'.format(self.episodes))  # Save the plot as an image
        # plt.clf()

        # smoothed_critic_loss = utils.smooth(losses_critic)
        # plt.plot(smoothed_critic_loss)
        # plt.xlabel('Episode')
        # plt.ylabel('Critic Loss')
        # plt.title('Critic Loss over Episodes')
        # plt.savefig('./train_results/ppo/smoothed_critic_loss_plot_{}.png'.format(self.episodes))  # Save the plot as an image
        # plt.clf()
        # print('Completed The Train')

    def evaluate(self,env):
        self.train(env)
        return_list = []
        for i_episode in range(self.num_evaluations):
            episode_return = 0
            state,info = env.reset()
            # print(state)
            # state = dic2state(state)
            done = False
            while not done:
                action,action_prob = self.agent.select_action(state)
                next_state, reward, terminal,truncted, _ = env.step(action)
                done = terminal or truncted
                # next_state = dic2state(next_state)
                state = next_state
                episode_return += reward
            # print(state)
            return_list.append(episode_return)
        return np.mean(return_list)