import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, prob, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, prob, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, prob, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, prob, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)
    



class Policy_network(nn.Module):

    def __init__(self, device, n_actions, batch_size, lr = 3e-4):
        super(Policy_network, self).__init__()
        self.device = device

        self.batch_size = batch_size

        self.actor = nn.Sequential(nn.Conv2d(3, 32, kernel_size = 8, stride = 4),
                                nn.ReLU(), nn.Conv2d(32, 64, kernel_size = 4, stride = 2),
                                nn.ReLU(), nn.Conv2d(64, 64, kernel_size = 3, stride = 1),
                                nn.ReLU(), nn.Flatten(), nn.Linear(4096, 256),
                                nn.ReLU(), nn.Linear(256, n_actions))
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr = lr)

    def forward(self, state):

        x = self.actor(state)
        x = F.softmax(x, dim=-1)
        x = torch.distributions.Categorical(x)

        return x
    
    def get_action(self, state):

        if state.shape[0] != self.batch_size:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        else:
            state = torch.tensor(state, dtype=torch.float).permute(0, 3, 1, 2).to(self.device)

        probs = self.forward(state)
        
        action = probs.sample()
        return action
        
class Value_network(nn.Module):

    def __init__(self, device, lr = 3e-4):
        super(Value_network, self).__init__()
        self.device = device

        self.control = nn.Sequential(nn.Conv2d(3, 32, kernel_size = 8, stride = 4),
                                nn.ReLU(), nn.Conv2d(32, 64, kernel_size = 4, stride = 2),
                                nn.ReLU(), nn.Conv2d(64, 64, kernel_size = 3, stride = 1),
                                nn.ReLU(), nn.Flatten(), nn.Linear(4096, 256),  
                                nn.ReLU(), nn.Linear(256, 1))
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr = lr)

    def forward(self, state):
        x = self.control(state)
        return x


class Policy(nn.Module):
    continuous = False # you can change this

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        self.device = device
        self.env = gym.make('CarRacing-v2', continuous = False)

        self.gamma = 0.99
        self.policy_clip = 0.2
        self.batch_size = 64
        self.N = 64
        self.n_epochs = 20
        self.gae_lambda = 0.95
        self.games = 10

        self.actor = Policy_network(device, batch_size = self.batch_size, n_actions=5)
        self.critic = Value_network(device)
        self.memory = ReplayBuffer(capacity=10000)

    def forward(self, x):
        # TODO
        
        action = self.act(x)

        if x.shape[0] != self.batch_size:
            x = torch.tensor(x, dtype=torch.float).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        else:
            x = torch.tensor(x, dtype=torch.float).permute(0, 3, 1, 2).to(self.device)

        policy = self.actor(x)
        value = self.critic(x)
        
        log_prob = torch.squeeze(policy.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, value, log_prob
    
    def act(self, state):
        # TODO
        action = self.actor.get_action(state)
        return action
    

    def learn(self):
        for _ in range(self.n_epochs):
            
            states, actions, old_probs, values, rewards, dones = \
                self.memory.sample(self.batch_size)

            advantages = np.zeros(rewards.shape, dtype=np.float32)
            print(rewards.shape)
            for t in range(len(rewards) - 1):
                discount = 1
                advantage_t = 0
                
                for i in range(t, len(rewards) - 1):
                    advantage_t += discount*(rewards[i] + self.gamma*values[i+1]*(1-dones[i]) \
                                            - values[i])
                    discount *= self.gamma*self.gae_lambda

                advantages[t] = advantage_t

            advantages = torch.tensor(advantages).to(self.actor.device)
            values = torch.tensor(values).to(self.actor.device)

            new_value = self.critic(states)
            new_log_prob = self.actor(states)
            
            prob_ratio = torch.exp(torch.tensor(new_log_prob - old_probs, dtype= torch.float))

            weights = advantages*prob_ratio
            clipped_weigths = torch.clamp(prob_ratio, 1-self.policy_clip, 
                                        1+self.policy_clip)*advantages
                
            actor_loss = -torch.min(weights, clipped_weigths).mean()

            returns = advantages + values
            print(advantages)
            critic_loss = nn.MSELoss(returns, new_value)

            total_loss = actor_loss + 0.5*critic_loss

            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            total_loss.backward()
            self.actor.optimizer.step()
            self.critic.optimizer.step()





    def train(self):
        # TODO
        n_steps = 0
        for _ in range(self.games):

            state, _ = self.env.reset()
            done = False
            
            while not done:
                n_steps += 1

                action, value, prob = self.forward(state)
                new_state, reward, done, _, _ = self.env.step(action)

                self.memory.push(state, action, prob, reward, new_state, done)

                if n_steps % self.N == 0:
                    self.learn()

                state = new_state




        return

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
