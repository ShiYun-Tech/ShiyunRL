import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from algorithms.ppo.model import Actor, Critic
from algorithms.ppo.utils import MemoryBuffer

class Agent():
    #state_dim, self.hidden_dim, action_dim, self.actor_lr, self.critic_lr, self.tau, self.gamma,self.capacity,self.device
    def __init__(self,state_dim,hidden_dim,action_dim,actor_lr,critic_lr,tau,gamma,capacity,batch_size,device):
        super(Agent, self).__init__()
        self.actor_net = Actor(state_dim,action_dim,hidden_dim)
        self.critic_net = Critic(state_dim,hidden_dim)
        self.buffer = MemoryBuffer(capacity)
        self.counter = 0
        self.training_step = 0
        self.gamma = gamma
        self.clip_param = 0.2
        self.max_grad_norm = 0.5
        self.ppo_update_time = 10
        self.batch_size = batch_size
        self.tau = tau
        self.device =device
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), actor_lr)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), critic_lr)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:,action.item()].item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_model(self):
        torch.save(self.actor_net.state_dict(), './train_results/ppo/actor.pt')
        torch.save(self.critic_net.state_dict(), './train_results/ppo/critic.pt')

    def load_model(self):
        self.actor_net.load_state_dict(torch.load('./train_results/ppo/actor.pt'))
        self.critic_net.load_state_dict(torch.load('./train_results/ppo/critic.pt'))

    def memory(self, state, action, a_log_prob,reward, next_state):
        self.buffer.add(state, action, a_log_prob,reward, next_state)

    def update(self):
        s_arr, a_arr,a_log_arr, r_arr, s1_arr = self.buffer.getAll()
        states = torch.tensor([state for state in s_arr], dtype=torch.float)
        actions = torch.tensor([action for action in a_arr], dtype=torch.int64).view(-1, 1)
        rewards = torch.tensor([reward for reward in r_arr], dtype=torch.float).view(-1, 1)
        old_action_log_probs = torch.tensor([a_log_prob for a_log_prob in a_log_arr], dtype=torch.float).view(-1, 1)
        next_states = torch.tensor([state for state in s1_arr], dtype=torch.float)
        value_loss_mean = 0.
        action_loss_mean = 0.
        R = 0
        Gt = []
        # print(rewards)
        rewards_flipped = rewards.flip(dims=[0])  # 使用 flip 函数来反转张量
        for r in rewards_flipped:
            # print("r:{}".format(r))
            R = r.item() + self.gamma * R  # `.item()` 可以从标量张量中获得Python数值
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)

        #print("The agent is updateing....")
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer.len)), self.batch_size, False):
                #with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(states[index])
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = self.actor_net(states[index]).gather(1, actions[index]) # new policy

                ratio = (action_prob/old_action_log_probs[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                # print("action_loss : ",action_loss)
                action_loss_mean = (action_loss_mean + action_loss.item())/2
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                #update critic network
                value_loss = F.mse_loss(Gt_index, V)
                # print("value_loss : ",value_loss)
                value_loss_mean = (value_loss_mean + value_loss.item())/2
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        self.buffer.clear()
        return action_loss_mean, value_loss_mean