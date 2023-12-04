import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ReplayBuffer:
    '''
    Class taken from the Lab code
    '''
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done, value, output):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, value, output)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = np.random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


class Critic_Network(nn.Module):
    def __init__(self, batch_size, device):
        super(Critic_Network, self).__init__()
        
        self.batch_size = batch_size
        self.device = device

        # Define all the layers 
        # Convolution since working with image
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 8, dilation = 4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 6, dilation = 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size = 3, dilation = 1)

        # Fully connected layers
        self.lin1 = nn.Linear(64*56*56, 256)
        self.lin2 = nn.Linear(256, 1)
    
        self.flat = nn.Flatten()

        # Put it all together in sequential
        self.net = nn.Sequential(self.conv1, nn.ReLU(), self.conv2, nn.ReLU(),
                                 self.conv3, nn.ReLU(), self.flat, self.lin1,
                                 nn.ReLU(), self.lin2, nn.ReLU()).to(device)
        
        
    def forward(self, state):
        # To apply the network I have to make sure I have torch tensors
        if state.shape[0] == self.batch_size:
            state = torch.tensor(state, dtype = torch.float).permute(0,3,1,2)
        else:
            state = torch.tensor(state, dtype = torch.float).permute(2,0,1).unsqueeze(0)
        
        # If possible move to GPU
        state = state.to(self.device)
        # Apply the network 
        state = self.net(state)

        return state
        

class Actor_Network(nn.Module):
    def __init__(self, batch_size, num_actions, device):
        super(Actor_Network, self).__init__()
        
        self.device = device
        self.num_actions = num_actions
        self.batch_size = batch_size

        # Define all the layers 
        # Convolution since working with image
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 8)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 4, stride = 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size = 2, stride = 4)

        # Fully connected layers
        self.lin1 = nn.Linear(64*56*56, 256)
        self.lin2 = nn.Linear(256, num_actions)
    
        self.flat = nn.Flatten()

        # Put it all together in sequential
        self.net = nn.Sequential(self.conv1, nn.ReLU(), self.conv2, nn.ReLU(),
                                 self.conv3, nn.ReLU(), self.flat, self.lin1,
                                 nn.ReLU(), self.lin2, nn.ReLU()).to(device)


        
    def forward(self, state):
        # To apply the network I have to make sure I have torch tensors
        if state.shape[0] == self.batch_size:
            state = torch.tensor(state, dtype = torch.float).permute(0,3,1,2)
        else:
            state = torch.tensor(state, dtype = torch.float).permute(2,0,1).unsqueeze(0)
        
        # If possible move to GPU
        state = state.to(self.device)
        # Apply the network 
        state = self.net(state)
        return state

    
    def get_action(self, state):
        # Apply the netwok
        state = self.forward(state)
        # Get the probability for each of the actions
        probs = F.softmax(state, dim = -1).detach().numpy()
        # sample the action
        action = np.random.choice(range(self.num_actions), 1, p = probs)
        return action

class Policy(nn.Module):
    continuous = False # you can change this

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        self.device = device
        self.env = gym.make('CarRacing-v2', continuous = False)
        self.num_actions = self.env.action_space.n

        self.lr = 1e-4
        self.num_iterations = 1
        self.batch_size = 10

        # Define the networks
        self.critic = Critic_Network(self.batch_size, device)
        self.actor = Actor_Network(self.batch_size, self.num_actions, device)

        self.actor_opt = torch.optim.RMSprop(self.actor.parameters())
        self.critic_opt = torch.optim.RMSprop(self.critic.parameters())

        self.replay_buffer = ReplayBuffer(capacity = 1e5)


    def calculate_gae(rewards, values, dones, value_next, done_next, gamma=0.99):

        values = torch.cat(values, value_next)
        dones = torch.cat(dones, done_next)

        # Calculate the delta values
        deltas = [r + (1-done)*gamma*v_next - v for r, v, v_next, done in zip(rewards, values, values[1:], dones)]
        
        # Calculate the advantages
        advantages = []
        adv = 0
        for delta in reversed(deltas):
            adv = delta + gamma * adv
            advantages.append(adv)
        advantages.reverse()
        
        # Calculate the target values for the value function
        returns = [adv + value for adv, value in zip(advantages, values)]
        
        # Convert lists to numpy arrays
        advantages = np.array(advantages, dtype=np.float32)
        returns = np.array(returns, dtype=np.float32)
        
        return advantages, returns

    def optimizers_step(self, batch_size ,gamma=0.99):
        state, action, reward, next_state, done, value, output = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(np.float32(done))
        value = torch.FloatTensor(value)
        output = torch.FloatTensor(output)

        # Next step
        next_action_output, next_value = self.forward(next_state[-1])
        next_action = self.act(next_state[-1])
        next_state, reward, next_done, truncated, _= self.env.step(next_action)

        advantages, returns = self.calculate_gae(reward, value, done, next_value, next_done)









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
        losses_ep = []
        for step in range(self.num_iterations):
            
            
            action = self.actor.get_action(state)
            action_output, predicted_value = self.forward(state)
            next_state, reward, terminated, truncated, _= self.env.step(action)
            #if truncated: print("!!! truncated")
            done = terminated
            
            self.replay_buffer.push(state, action,  reward, next_state, terminated)
            if len(self.replay_buffer) > self.batch_size:
                loss = self.optimizers_step(self.batch_size)
                losses_ep.append(loss)
            
            state = next_state
            episode_reward += reward
            
            if done:
                state = self.env.reset()

        

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
