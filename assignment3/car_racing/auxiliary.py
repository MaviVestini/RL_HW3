import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet50


class Network(nn.Module):
    '''
    Class used to define the network structure
    '''

    def __init__(self, output_size = 1, input_channels = 4, lr = 1e-4,
                device=torch.device('cpu')):
        super(Network, self).__init__()

        # Number of channels of the input
        self.input_size = input_channels

        # Define the networks
        '''
        Both the network have structure: 3x(Conv+ReLU+MaxPoo) + 2x(Linear)
        '''
        self.net = nn.Sequential(
                        # Convolution
                        nn.Conv2d(self.input_size, 16, kernel_size=4, stride=2),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(16),
                        nn.MaxPool2d(kernel_size=2, stride=2),

                        nn.Conv2d(16, 32, kernel_size=2, stride=1),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(32),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        
                        nn.Conv2d(32, 64, kernel_size=2, stride=1),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(64),
                        nn.MaxPool2d(kernel_size=2, stride=2),

                        # Linear
                        nn.Flatten(),
                        nn.Linear(1024, 256),
                        nn.ReLU(inplace=True),
                        nn.Linear(256, output_size)
                    ).to(device)
        
        # Initialize weights
        self.init_weights()

        # Define the optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)


    def init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        '''
        forward of the network given the input element x
        '''

        if torch.is_tensor(x) and x.shape[0] == self.input_size:
            # x is a tensor but it doesn't have the batch dimension
            x = x.unsqueeze(0) 
        elif x.shape[0] == self.input_size:
            # x is not a tensor and also is missing the batch dimension
            x = torch.FloatTensor(x).unsqueeze(0)
        elif not torch.is_tensor(x):
            # x is of the right shape but not a tensor
            x = torch.FloatTensor(x)

        # x is a tensor of the right size, pass it through the network
        output = self.net(x)
        # return the output
        return output


class ReplayBuffer():

    def __init__(self, capacity, alpha, beta, batch_size):
        
        self.capacity = capacity
        self.beta = beta
        self.beta_increment = 0.0001
        self.alpha = alpha 
        self.buffer = np.empty(self.capacity, dtype=object)
        self.position = 0
        self.priorities = np.zeros(self.capacity)
        self.batch_size = batch_size
        

    def push(self, state, action, rewards, next_state, done):
        '''
        Add an element to the buffer, two possibilities: 
        1. buffer is full -> drop the sample with the least priority
        2. not full -> add the new experience (state, action, rewars, next_state)
        '''

        # find the priority for the new experience
        # I set the priority to 1 if the buffer is empty, else the maximum
        priority = 1 if self.position == 0 else np.max(self.priorities)

        if self.position != self.capacity - 1: 
            # There is still space in the buffer 
            # Add the new element to the lists
            self.priorities[self.position] = priority
            self.buffer[self.position] = (state, action, rewards, next_state, done)
            # Move the position
            self.position += 1
        else:
            # Buffer is full, then indentify the element with the least 
            # priority and swap it with the new one
            idx_least = np.argmin(self.priorities)
            # Add the new element in the index correspondind to the 
            # element with least priority
            self.priorities[idx_least] = priority
            self.buffer[idx_least] = (state, action, rewards, next_state, done)

    def update_priorities(self, indices, new_priorities):
        '''
        After computing the new priorities this function modify the priority
        list inserting them 
        '''
        self.priorities[indices] = new_priorities

    def get_samples(self):
        '''
        sample batch_size elements according to the normalized priorities
        and compute the importance sampling weights as described in 
        https://arxiv.org/pdf/1511.05952.pdf#section.3
        '''
        
        # extract the priorities of the stored elements
        priorities = self.priorities[:self.position]
        # elevated to the alpha
        priorities = priorities**self.alpha
        # normalize
        probability = priorities/np.sum(priorities)

        # Sample using the computed probabilities
        sample_idx = np.random.choice(self.position, self.batch_size, replace=True, p = probability)
        samples = self.buffer[sample_idx]
        

        # importance weights
        n_elements_in_buffer = self.position + 1
        weights = (n_elements_in_buffer*probability[sample_idx])**(-self.beta)
        # Between 0 and 1
        weights = weights/np.max(weights)
        
        self.update_beta()

        return samples, torch.FloatTensor(weights), sample_idx
    
    def update_beta(self):
        self.beta = min(1, self.beta + self.beta_increment)





