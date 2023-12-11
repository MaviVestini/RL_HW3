import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from auxiliary import Network, ReplayBuffer


class Policy(nn.Module):
    continuous = False # you can change this

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        self.device = device        

        # Define all the variables
        self.replay_period = 128
        self.minibatch = 32
        self.buffer_size = 10000
        self.budget = 500_000

        # Add some exploration/exploitation using epsilon
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.target_epsilon = 0.01
        self.use_epsilon = False

        # Initial exploration don't change epsilon for the first steps
        self.exploration_start = 25_000

        # For the TD-error
        self.gamma = 0.9

        # Define the replay buffer
        self.buffer = ReplayBuffer(self.buffer_size, alpha = 0.7, 
                                beta = 0.5, batch_size = self.minibatch)

        # Define the environment in a discrete space
        self.env = gym.make('CarRacing-v2', continuous = False)#, render_mode='human')
        self.n_actions = self.env.action_space.n

        # Keep track of the last 4 frames to stack them together tp use 
        # as input to the network
        self.state_stacked = [torch.zeros((84, 84)) for _ in range(4)]

        # Initialize both the target and policy networks
        self.q_net = Network(output_size=self.n_actions, lr=1e-5)
        self.target_net = Network(output_size=self.n_actions)

        # To try and make the problem simpler I'll transform all the images to gray scale
        self.gray = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                    torchvision.transforms.Grayscale(),
                                                    torchvision.transforms.ToTensor()])


    def forward(self, x):
        # TODO
        '''
        Pass x in the qnetwork and get the output
        '''
        predicted_q = self.q_net(x)
        return predicted_q
    

    def act(self, state):
        # TODO
        '''
        Given the state choose an action to preform. \n
        Note that. If we are in the evaluation phase, I don't want to 
        use epilon but just take the actions from the network, instead if 
        we are training I want to use epsilon
        '''

        # If we are not using epsilon this means we are in the evaluation phase
        # and I need to stack the last frames to have the correct state for 
        # the network
        if not self.use_epsilon:
            # Transform and append
            self.state_stacked = self.state_stacked[1:] + \
                                    [self.preprocess_states(state).squeeze()]
            # Stack into a tensor
            state = torch.stack(self.state_stacked)
            # Add batch dimension for the network
            state = torch.tensor(state).float().unsqueeze(0)

            # take action according to the network
            predicted_q = self.forward(state)
            action = torch.argmax(predicted_q).item()

            return action

        # If we are training and I sample a number under epsilon 
        if np.random.random() < self.epsilon:
            # take a random action
            action = np.random.choice(self.n_actions)
        else:
            # take action according to the network
            predicted_q = self.forward(state)
            action = torch.argmax(predicted_q).item()
        
        # return the action
        return action
    
    
    def preprocess_states(self, state):
        '''
        Input: tensor (96,96,3)
        Output: tensor (1, 84, 84), obtained by changing the image to 
        black and white and cropping it to take out the bottom part 
        '''
        # Crop the bottom of the picture
        state = state[:84, 6:90,]

        # Change to black and white
        b_w = self.gray(state)
        return b_w


    def learn(self, time):
        '''
        Function that perform performs the parameters update
        '''
        # Get the samples and importance sampling weights
        samples, weights, sample_idx = self.buffer.get_samples()

        # Extract 
        state = torch.FloatTensor([sample[0].numpy() for sample in samples])
        action = torch.tensor([sample[1] for sample in samples], dtype=torch.int64)
        rewards = torch.FloatTensor([sample[2] for sample in samples])
        next_state = torch.FloatTensor([sample[3].numpy() for sample in samples])
        done = torch.IntTensor([sample[4] for sample in samples])

        # TD-error: 
        # δ_j = R_j + γ_j Qtarget(Sj , argmax_a (Q(S_j, a))) − Q(S_j−1, A_j−1)
        # Get the outputs for the next_state of both the target and q network 
        with torch.no_grad():
            # Output of target given next_states
            target_output = self.target_net(next_state)
            # Output of q_network given next_states for argmax_a (Q(S_j, a))if
            q_output_next = self.forward(next_state)
            # Output of q_network given state for Q(S_j−1, A_j−1)
            forward_output = self.forward(state)

        # Select the action for the target network with the argmax of the q_network
        actions_for_traget = torch.argmax(q_output_next, dim=1)

        # Qtarget(Sj , argmax_a Q(Sj , a))
        Q_target = torch.gather(target_output, 1, actions_for_traget.unsqueeze(1))

        # mask the values corresponding to terminal states
        mask = 1-done.unsqueeze(1)
        Q_target = Q_target*mask

        # Q(S_j−1, A_j−1)
        Q_output =  torch.gather(forward_output, 1, action.unsqueeze(1))

        # Compute the deltas
        delta = rewards.unsqueeze(1) + self.gamma * Q_target - Q_output
        
        # Update the priorities pj ← |δj |
        priorities = abs(delta.numpy())
        self.buffer.update_priorities(sample_idx, priorities.reshape(-1) + 1e-6)
        
        # Compute the loss and perform Optimizer step
        loss = torch.mean((weights*delta)**2).requires_grad_(True)
        self.q_net.optimizer.zero_grad()
        loss.backward()      
        self.q_net.optimizer.step()
        
        # once every 3 times I compute the loss let's update the target
        # To finish copy the new network weights to the target one
        self.target_net.load_state_dict(self.q_net.state_dict())
    
        return loss.detach().numpy()

    def train(self):
        # TODO
        self.use_epsilon = True

        # Initialize the env
        self.reset_env()

        # get the initial state by stacking the frames
        state = torch.stack(self.state_stacked, dim = 0)
        episode_reward = []
        loss = []

        # Start the iteration
        for t in range(self.budget):

            # Select the action 
            action = self.act(state)
            # perform the action and orserve the output
            new_observation, reward, done, truncated, _ = self.env.step(action)
            episode_reward.append(reward)

            # Given the new observation find the new state
            self.state_stacked = self.state_stacked[1:] + \
                            [self.preprocess_states(new_observation).squeeze()]
            new_state = torch.stack(self.state_stacked, dim = 0)

            # Store everything in the buffer
            self.buffer.push(state, action, reward, new_state, done)

            # check if it's time to learn
            if t % (self.replay_period + 1) == 0:
                loss.append(self.learn(t))

                if t > self.exploration_start:
                    # The "just use random" initial period is done and I can 
                    # update epsilon
                    self.epsilon = max(self.epsilon_decay*self.epsilon, 
                                    self.target_epsilon)
            

            if not done and not truncated:
                state = new_state
            else:
                print('reward:', np.sum(episode_reward), 
                    '|| mean loss:', np.mean(loss), 
                    '|| epsilon:', self.epsilon, 
                    '|| steps:', len(episode_reward), 
                    '|| iteration', t)
                
                if np.sum(episode_reward) > 100:
                    self.save(np.mean(episode_reward))

                # Initialize the env
                self.reset_env()

                # get the initial state by stacking the frames
                state = torch.stack(self.state_stacked, dim = 0)
                episode_reward = []
                loss = []
        return


    def reset_env(self):
        '''
        Reset the environment in case of terminal state and skip the first
        60 frames since they are just the zoom in part
        '''

        # Initialize the environment
        state, _ = self.env.reset() 
        # List to store the last 4 states since I want to give some temporal 
        # informations to the network
        self.state_stacked = []

        # I noticed that the first 50/60 frames are the camera zooming in
        for i in range(60):
            # Stay put during the zoom-in phase
            output = self.env.step(0)
            # Store the B&W representation of the last 4 frames
            if i in [56, 57, 58, 59]:
                self.state_stacked.append(self.preprocess_states(output[0]).squeeze())
        
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