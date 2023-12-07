import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import wandb


wandb.init(
    entity='vestini-1795724', 

    # set the wandb project where this run will be logged
    project="car_racing",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 3e-7,
    "env": 'CarRacing-v2',
    "epochs": 40,
    }
)


class Policy(nn.Module):
    continuous = False # you can change this
    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        self.device = device        

        super(Policy, self).__init__()
        self.device = device

        # Define the environment
        self.env = gym.make('CarRacing-v2', continuous = False)#, render_mode='human')
        n_actions = self.env.action_space.n

        # Assign the constants
        self.gamma = 0.8
        self.tau = 0.7
        self.lr = 0.00001
        self.n_episodes = 100
        self.n_steps = 256
        self.last_k = 10
        self.state_stacked = [torch.zeros((96, 96)) for _ in range(4)]

        # Define the networks
        self.actor = nn.Sequential(
                        nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),

                        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),

                        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),

                        nn.Flatten(),
                        nn.Linear(64 * 12 * 12, 128),
                        nn.ReLU(inplace=True),
                        nn.Linear(128, n_actions)
                    )

                                
        self.critic = nn.Sequential(
                        nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),

                        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),

                        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),

                        nn.Flatten(),
                        nn.Linear(64 * 12 * 12, 128),
                        nn.ReLU(inplace=True),
                        nn.Linear(128, 1)
                    )
        

        self.gray = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                    torchvision.transforms.Grayscale(),
                                                    torchvision.transforms.ToTensor()])
        
        # And the optimizers
        #self.optimizer = torch.optim.RMSprop(self.parameters(), lr = 0.00001)
        self.optimizer_A = torch.optim.RMSprop(self.actor.parameters(), lr = 3e-4)
        self.optimizer_C = torch.optim.RMSprop(self.critic.parameters(), lr = 3e-3)

    def forward(self, x):
        # TODO

        if x.shape[0] != self.n_episodes:
            if torch.is_tensor(x):
                x = x.unsqueeze(0)
            else:
                x = torch.tensor(x).float().unsqueeze(0)
            
        else:
            x = torch.tensor(x).float()
    
        # Compute value and get the action using the networks
        value = self.critic(x)
        action = self.act(x)
        probs = F.softmax(self.actor(x), dim = -1)
        
        return value, action, probs
    
    def act(self, state):
        # TODO
        eval = False

        if not torch.is_tensor(state):
            eval = True
            self.state_stacked = self.state_stacked[1:] +  [self.gray(state).squeeze()]
            state = torch.stack(self.state_stacked)
            state = torch.tensor(state).float().unsqueeze(0)

        # Apply the network
        actions = self.actor(state)
        # Get a distribution for the actions
        distr = torch.distributions.Categorical(F.softmax(actions, dim=-1))
        # Sample and return the action 
        if not eval:
            return distr.sample()

        return distr.sample().detach().numpy().item()
        

    def train(self):
        # TODO
        tot_reward = 0
        state, _ = self.env.reset() 
        entropy = 0
        self.state_stacked = []

        for i in range(50):
            state,_,_,_,_ = self.env.step(0)
            if i in [46, 47, 48, 49]:
                self.state_stacked.append(self.gray(state).squeeze())

        for epoch in range(self.n_episodes):
            
            # Arrays for the informations that will have to be stored
            actions = np.zeros((self.n_steps,), dtype=np.int64)
            dones = np.zeros((self.n_steps,), dtype=bool)
            rewards = np.zeros((self.n_steps,), dtype=float)
            values = np.zeros((self.n_steps,), dtype=float)
            states = np.zeros((self.n_steps, 4, 96, 96), dtype=float)
            log_probs = torch.zeros((self.n_steps,), dtype=float)
            episode_reward = 0
            steps_into_ep = 0
            negative_patience = 50

            # Rollout Phase
            for i in range(self.n_steps):
                steps_into_ep += 1
                
                # Store the outputs of the networks
                states[i] = torch.stack(self.state_stacked, dim=0)
                values[i], actions[i], prob = self.forward(states[i])
                
                log_probs[i] = torch.log(prob[0][actions[i]])

                state, reward, done, truncated, _ = self.env.step(actions[i])
                self.state_stacked = self.state_stacked[1:] +  [self.gray(state).squeeze()]

                rewards[i] = reward
                dones[i] = done

                entropy -= torch.sum(torch.mean(prob) * torch.log(prob))


                # note that actions: {0: stay still, 1: right, 2: left, 3: gas, 4: brake}
                if reward > 0 and actions[i] == 3:
                    # I want to give better reward for the acceleration
                    rewards[i] = 10

                # the negative patience is used to reward with an high negative value
                # the agent is it steps too much in the grass (not in the street)
                if rewards[i] < 0:
                    negative_patience -= 1
                else:
                    negative_patience = 50

                if negative_patience == 0:
                    rewards[i-steps_into_ep:i] = -10 
                    dones[i] = True

                

                # Penalize going on the grass
                first_to_consider = max(0, i-self.last_k)
                if (rewards[first_to_consider:i+1] < 0).all():
                    rewards[first_to_consider:i+1] = -0.5

                episode_reward += rewards[i]


                if dones[i]:
                    steps_into_ep = 0
                    self.state_stacked = []
                    episode_reward = 0
                    steps_into_ep = 0
                    negative_patience = 50
                    tot_reward += episode_reward
                    dones[i] = 1
                    state, _ = self.env.reset() 
                    for j in range(50):
                        state,_,_,_,_ = self.env.step(0)
                        if j in [46, 47, 48, 49]:
                            self.state_stacked.append(self.gray(state).squeeze())

        

            # Learning Phase
            next_state = torch.stack(self.state_stacked, dim=0)
            next_val, _, _ = self.forward(next_state)

            #update actor critic
            values = torch.FloatTensor(values)
            dones = torch.BoolTensor(dones-1)
            rewards = torch.FloatTensor(rewards)
            next_values = torch.concat([values[1:], torch.FloatTensor([next_val])])

            advantage = self.compute_advantages(rewards, values, next_values, gamma=0.99)
            
            actor_loss = (-log_probs * advantage).mean().requires_grad_(True)
            actor_loss += 0.0001 * entropy
            critic_loss = 0.5 *advantage.pow(2).mean().requires_grad_(True)
            
            a2c_loss = actor_loss + 0.5*critic_loss
            
            self.optimizer_A.zero_grad()
            actor_loss.backward()
            self.optimizer_A.step()

            self.optimizer_C.zero_grad()
            critic_loss.backward()
            self.optimizer_C.step()

            wandb.log({"batch_reward": episode_reward, "loss_actor": actor_loss, "loss_critic": critic_loss, "loss": a2c_loss})

            print(f'epoch: {epoch}, loss: {a2c_loss}')
            print(np.unique(actions, axis=0, return_counts=True))
            self.save(epoch)

        return


    def compute_advantages(self, rewards, values, next_values, gamma=0.99):
        advantages = torch.zeros_like(rewards)
        advantage = 0

        # Iterate in reverse to calculate advantages backwards
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_values[t] - values[t]
            advantage = delta + gamma * 0.95 * advantage  # GAE parameter (0.95)

            advantages[t] = advantage

        

        return advantages

    
    def save(self, iter = None):
        if iter:
            torch.save(self.state_dict(), f'model_{iter}.pt')
        else:
            torch.save(self.state_dict(), f'model.pt')
    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))
    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret