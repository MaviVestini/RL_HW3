import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class Critic_Network(nn.Module):
    def __init__(self, batch_size, device):
        super(Critic_Network, self).__init__()
        
        self.batch_size = batch_size
        self.device = device

        # Define all the layers 
        # Convolution since working with image
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)

        # Fully connected layers
        self.lin1 = nn.Linear(4096, 256)
        self.lin2 = nn.Linear(256, 1)
    
        self.flat = nn.Flatten()

        # Put it all together in sequential
        self.net = nn.Sequential(self.conv1, nn.ReLU(), self.conv2, nn.ReLU(),
                                self.conv3, nn.ReLU(), self.flat, self.lin1,
                                nn.ReLU(), self.lin2).to(device)
        
        
    def forward(self, state):
        # To apply the network I have to make sure I have torch tensors
        if state.shape[0] == self.batch_size:
            state = torch.tensor(state, dtype = torch.float).permute(0,3,1,2)
        else:
            state = torch.tensor(state, dtype = torch.float).clone().detach().permute(2, 0, 1).unsqueeze(0)
        
        # If possible move to GPU
        state = state.to(self.device)
        # Apply the network 
        state = self.net(state)

        return state.detach().numpy()
        

class Actor_Network(nn.Module):
    def __init__(self, batch_size, num_actions, device):
        super(Actor_Network, self).__init__()
        
        self.device = device
        self.num_actions = num_actions
        self.batch_size = batch_size

        # Define all the layers 
        # Convolution since working with image
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)

        # Fully connected layers
        self.lin1 = nn.Linear(4096, 256)
        self.lin2 = nn.Linear(256, num_actions)
    
        self.flat = nn.Flatten()

        # Put it all together in sequential
        self.net = nn.Sequential(self.conv1, nn.ReLU(), self.conv2, nn.ReLU(),
                                self.conv3, nn.ReLU(), self.flat, self.lin1,
                                nn.ReLU(), self.lin2).to(device)


        
    def forward(self, state):
        # To apply the network I have to make sure I have torch tensors
        if state.shape[0] == self.batch_size:
            state = torch.tensor(state, dtype = torch.float).permute(0,3,1,2)
        else:
            state = torch.tensor(state, dtype = torch.float).clone().permute(2, 0, 1).unsqueeze(0)
        
        # If possible move to GPU
        state = state.to(self.device)
        # Apply the network 
        state = self.net(state)
        return state.detach().numpy()

    
    def get_action(self, state):
        # Apply the netwok
        state = self.forward(state)
        # Get the probability for each of the actions
        probs = F.softmax(torch.tensor(state), dim = -1).detach().numpy()[0]
        # sample the action
        action = np.random.choice(range(self.num_actions), 1, p = probs)[0]
        return action


class Policy(nn.Module):
    continuous = False # you can change this

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        self.device = device
        self.env = gym.make('CarRacing-v2', continuous = False)
        self.num_actions = self.env.action_space.n

        self.lr = 1e-6
        self.num_iterations = 100
        self.batch_size = 10

        # Define the networks
        self.critic = Critic_Network(self.batch_size, device)
        self.actor = Actor_Network(self.batch_size, self.num_actions, device)

        self.actor_opt = torch.optim.RMSprop(self.actor.parameters(), lr = self.lr)
        self.critic_opt = torch.optim.RMSprop(self.critic.parameters(), lr = self.lr)
        

    def forward(self, x):
        # TODO
        actor = self.actor(x)
        critic = self.critic(x)
        return actor, critic
    
    def act(self, state):
        # TODO
        return self.actor.get_action(state)

    def train(self):
        state, _ = self.env.reset()
        episode_reward = 0
        losses = []

        for step in range(self.num_iterations):

            actions = np.zeros((self.batch_size,), dtype=np.int32)    
            states = np.zeros((self.batch_size,) + self.env.observation_space.shape, dtype=float)
            dones = np.zeros((self.batch_size,), dtype = bool)
            values = np.zeros((self.batch_size,), dtype = bool)
            rewards = np.zeros((self.batch_size,), dtype = bool)
            outputs =  np.zeros((self.batch_size,) + self.env.action_space.shape, dtype=float)
            
            for episode in range(self.batch_size):

                states[episode] = state
                actions[episode] = self.actor.get_action(state)
                outputs[episode], values[episode] = self.forward(state)
                
                next_state, rewards[episode], dones[episode], truncated, _= self.env.step(actions[episode])
                #if truncated: print(!!! truncated)

                state = next_state
                episode_reward += rewards[episode]
                
                if dones[episode]:
                    state = self.env.reset()
            

            _, value_next =  self.forward(state)
            # Optimizers and losses
            advantages, returns = self.compute_adv(rewards, values, dones, value_next)
            print(advantages, returns)

    
    def compute_adv(self, rewards, values, dones, value_next, gamma=0.99):
        values = np.appens(values, value_next)

        # Calculate the delta values
        deltas = [r + (1-done)*gamma*v_next - v for r, v, v_next, done in zip(rewards, values, values[1:], dones)]
        
        # Calculate the advantages
        advantages = np.zeros_like(rewards)
        adv = 0
        for i in reversed(range(len(deltas))):
            adv = deltas[i] + gamma * adv
            advantages[i] = adv
            
        advantages = np.flip(advantages)
        # Calculate the target values for the value function
        returns = advantages + values[:-1]

        return advantages, returns
        

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret