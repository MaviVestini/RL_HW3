import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
#import wandb


"""
# Used wand to keep track of the training
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
"""

class Policy(nn.Module):
    continuous = False # you can change this

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        self.device = device        

        # Define the environment in a discrete space
        self.env = gym.make('CarRacing-v2', continuous = False, render_mode='human')
        # Find the possible number of actions
        n_actions = self.env.action_space.n

        # Assign the constants
        
        # For the advantages
        self.gamma = 0.9

        # For the training
        self.n_epochs = 500
        self.batch_size = 256

        # Some needed variables 
        # To assign custom reward
        self.last_k = 50
        self.state_stacked = [torch.zeros((84, 84)) for _ in range(4)]

        # Define the networks
        '''
        Both the network have structure: 3x(Conv+ReLU+MaxPoo) + 2x(Linear)
        for the actor network I'll apply softmax at the end
        '''
        self.actor = nn.Sequential(
                        # Convolution
                        nn.Conv2d(4, 16, kernel_size=4, stride=2),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),

                        nn.Conv2d(16, 32, kernel_size=2, stride=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        
                        nn.Conv2d(32, 64, kernel_size=2, stride=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),

                        # Linear
                        nn.Flatten(),
                        nn.Linear(1024, 256),
                        nn.ReLU(inplace=True),
                        nn.Linear(256, n_actions),

                        nn.Softmax(dim = -1)
                    ).to(device)
        

        
        self.critic = nn.Sequential(
                        # Convolution
                        nn.Conv2d(4, 16, kernel_size=4, stride=2),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),

                        nn.Conv2d(16, 32, kernel_size=2, stride=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        
                        nn.Conv2d(32, 64, kernel_size=2, stride=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),

                        # Linear
                        nn.Flatten(),
                        nn.Linear(1024, 256),
                        nn.ReLU(inplace=True),
                        nn.Linear(256, 1)
                    )
        
        # To try and make the problem simpler I'll transform all the images to gray scale
        self.gray = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                    torchvision.transforms.Grayscale(),
                                                    torchvision.transforms.ToTensor()])
        
        # The optimizer
        self.optimizer_A = torch.optim.RMSprop(self.actor.parameters(), lr = 1e-5)
        self.optimizer_C = torch.optim.RMSprop(self.critic.parameters(), lr = 3e-5)

    def forward(self, x):
        '''
        Given the state: that for me will be the last 4 b&w images stacked
        Returns: The outputs of the network + the action to be taken 
        '''
        # TODO

        # x is the np array or tensor with size (4, 96, 96) so this 
        # means that it's just one state and I have to add the batch dimension
        if torch.is_tensor(x):
            # x is already a tensor so I don't need to change the type
            x = x.unsqueeze(0)
        else:
            # change both type and size
            x = torch.tensor(x).float().unsqueeze(0)
        
        # Pass x through the networks and extract the action

        # predicted value
        value = self.critic(x)
        # predicted probabilities for all the actions 
        probs = self.actor(x)
        # sample an action
        action = torch.distributions.Categorical(probs).sample()
        
        # Return
        return value, action, probs
    
    def act(self, state):
        '''
        Function used for the evaluation
        Given the state: one single RGB frame 
        Returns: the action to be taken using the actor network output
        '''
        # TODO
        
        # The networks have been trained by taking as input the last 4 
        # b&w images, so I will transform the frame in B&W and stack it
        # with the previous frames
        
        # Transform and append
        self.state_stacked = self.state_stacked[1:] +  [self.preprocess_states(state).squeeze()]
        # Stack into a tensor
        state = torch.stack(self.state_stacked)
        # Add batch dimension for the network
        state = torch.tensor(state).float().unsqueeze(0)

        # Forward in the actor network
        distr = torch.distributions.Categorical(self.actor(state))
        
        # sample and return the action
        return distr.sample().detach().numpy().item()
    
    def preprocess_states(self, state):
        '''
        Input: tensor (96,96,3)
        Output: tensor (1, 84, 84), obtained by changing the image to 
        black and white and cropping it to take out the bottom part 
        '''
        # Crop the bottom of the picture
        state = state[:84,6:90,]
        
        # Change to black and white
        b_w = self.gray(state)
        return b_w



    def compute_advantages(self, rewards, dones, values, next_value):
        '''
        Input: Informations seen during the rollout phase: \n
            - rewards: tensor with the rewards obtained
            - dones: tensor with 0s and 1s to mark a terminal state
            - values: tensor that contains the predicted values
            - next_value: predicted value given the last seen state
        
        Output: the advantages computed using TD errors
        '''

        # Initialization of the returns, last value equal to the last predicted value
        returns = np.append(np.zeros_like(rewards), [next_value], axis=0)
    
        # Iterating in reverse
        for t in reversed(range(rewards.shape[0])):
            # Compute the returns
            returns[t] = rewards[t] + self.gamma * returns[t + 1]*(1 - dones[t])
        
        # Compute the advantages remembering that Adv = returns - values
        advantages = torch.FloatTensor(returns[:-1]) - values

        # Return the advantages
        return advantages


    def train(self):
        '''
        Main function to train the networks. For each epoch (iteration) I'll:\n
            - Initialize the vectors
            - Perform the Rollout Phase in which I store all the env outputs
            - Perform the Learning Phase to optimize the networks
        '''
        # TODO

        # Initialize the environment
        state, _ = self.env.reset() 
        # List to store the last 4 states since I want to give some temporal 
        # informations to the network
        self.state_stacked = []
        episode_reward = 0

        # I noticed that the first 50/60 frames are the camera zooming in
        for i in range(60):
            # Stay put during the zoom-in phase
            output = self.env.step(0)
            # Store the B&W representation of the last 4 frames
            if i in [56, 57, 58, 59]:
                self.state_stacked.append(self.preprocess_states(output[0]).squeeze())

        
        for epoch in range(self.n_epochs):

            # Arrays for the informations that will have to be stored
            actions = np.zeros((self.batch_size,), dtype=np.int64)
            dones = np.zeros((self.batch_size,), dtype=np.int64)
            rewards = np.zeros((self.batch_size,), dtype=float)
            values = np.zeros((self.batch_size,), dtype=float)
            states = np.zeros((self.batch_size, 4, 84, 84), dtype=float)
            log_probs = torch.zeros((self.batch_size,), dtype=float)
            entropies = torch.zeros((self.batch_size,), dtype=float) 


            # Rollout Phase
            for i in range(self.batch_size):

                # Stack the last 4 frames
                states[i] = torch.stack(self.state_stacked, dim=0)
                # forward with the network to get the predicted value, action 
                # and probabilities of each action
                values[i], actions[i], prob = self.forward(states[i])
                # Compute the logarithm of the probability of the selected action
                log_probs[i] = torch.log(prob[0][actions[i]])

                # Perform the action and observe the state and reward
                state, reward, done, truncated, _ = self.env.step(actions[i])
                # Transform the new state in b&w and store it together with 
                # the previous states
                self.state_stacked = self.state_stacked[1:] +  [self.preprocess_states(state).squeeze()]
                # Compute the entropy
                entropy = - torch.sum(torch.mean(prob) * torch.log(prob))

                # Store the new informations in the arrays
                rewards[i] = reward
                dones[i] = done
                entropies[i] = entropy

                # If the episode has finished reset the env and the episode 
                # related variables
                if dones[i] or truncated:
                    # Variables
                    self.state_stacked = []
                    episode_reward = 0

                    # Environment
                    state, _ = self.env.reset() 
                    for j in range(60):
                        obs = self.env.step(0)
                        if j in [56, 57, 58, 59]:
                            self.state_stacked.append(self.preprocess_states(obs[0]).squeeze())


            # Learning Phase

            # Compute the predicted next value given the current network
            next_state = torch.stack(self.state_stacked, dim=0)
            with torch.no_grad():
                next_val, _, _ = self.forward(next_state)

            # Transform everything to tensors
            values = torch.FloatTensor(values)
            dones = torch.IntTensor(dones)
            rewards = torch.FloatTensor(rewards)

            # Compute advantages
            advantages = self.compute_advantages(rewards, dones, values, next_val)
            
            # Compute the loss: Advanced actor critic loss
            # actor loss = - log(\pi)*advantages
            actor_loss = -(-log_probs * advantages).mean().requires_grad_(True)
            # to which I sum the mean entropy
            actor_loss += 0.0001 * entropies.mean()
            # critic loss = mse(returns, values), but I can compute it 
            # as written below since Adv = returns - values 
            critic_loss = 0.5 *advantages.pow(2).mean().requires_grad_(True)
            
            # Total loss sum of actor loss and 0.5*critic 
            a2c_loss = actor_loss + 0.5*critic_loss

            # Perform the model update
            self.optimizer_A.zero_grad()
            actor_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
            self.optimizer_A.step()


            self.optimizer_C.zero_grad()
            critic_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
            self.optimizer_C.step()

            #wandb.log({"batch_reward": episode_reward, "loss_actor": actor_loss, "loss_critic": critic_loss, "loss": a2c_loss})

            #print(f'epoch: {epoch}, loss: {a2c_loss}')
            #print(np.unique(actions, axis=0, return_counts=True))
            


            del actions
            del dones
            del rewards 
            del values
            del states
            del log_probs
            del entropies

        return

    
    
    def save(self, it = None):
        if not it: torch.save(self.state_dict(), f'model.pt')
        else: torch.save(self.state_dict(), f'model_{it}.pt')
    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))
    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret