import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from algorithms.sac import utils
from algorithms.sac.sac import Agent

def sac_gene_test_runner(env, config):
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
            action = agent.select_action(state)
            next_state, reward, terminated,truncted, info = env.step(action)
            done = terminated or truncted
            state = next_state
            total_rewards += reward
            if done :
                break
        rewards.append(total_rewards)
    
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('SAC Reward over Episodes')
    plt.savefig('./test_results/sac/test_sac_best_gene_reward_plot.png')  # Save the plot as an image
    plt.clf()  # Clear the plot for the next update

class sacRunner(object):
    def __init__(self,state_dim,action_dim,hidden_dim,actor_lr,critic_lr,gamma,tau,seed,episodes,capacity,batch_size,device='cpu'):
        self.num_evaluations = 100
        self.device = device
        #gene
        self.hidden_dim = hidden_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = 1e-3
        self.target_entropy = 0.5  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.episodes = episodes
        self.capacity = capacity
        self.minimal_size = 10
        self.batch_size = batch_size
        self.seed = seed
        
        self.agent = Agent(state_dim, self.hidden_dim, action_dim, self.actor_lr, self.critic_lr, self.alpha_lr,
                    self.target_entropy, self.tau, self.gamma, self.device)
        
    def save_model(self):
        self.agent.save_model()
        print('Save model successfully')
        #save the gene to the txt file
        with open('./configs/sac_best_gene_config.yaml', 'w') as f:
            f.write("algo: sac\n")
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
        replay_buffer = utils.ReplayBuffer(self.capacity)
        return_list = utils.train_off_policy_agent(env, self.agent, self.episodes,
                                                    replay_buffer, self.minimal_size,
                                                    self.batch_size)
        episodes_list = list(range(len(return_list)))
        mv_return = utils.moving_average(return_list, 9)
        plt.plot(episodes_list, mv_return)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('SAC')
        plt.savefig('./train_results/sac/Average_Returns_{}.png'.format(self.episodes))
        plt.clf()

    def evaluate(self,env):
        self.train(env)
        return_list = []
        for ep in range(self.num_evaluations):
            episode_return = 0
            state,info = env.reset()
            # print(state)
            # state = dic2state(state)
            done = False
            while not done:
                action = self.agent.select_action(state)
                next_state, reward, terminal,truncted, _ = env.step(action)
                done = terminal or truncted
                # next_state = dic2state(next_state)
                state = next_state
                episode_return += reward
            # print(state)
            return_list.append(episode_return)
        return np.mean(return_list)