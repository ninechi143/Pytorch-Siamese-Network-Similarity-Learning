import torch
import torch.nn as nn
import numpy as np


class FeatureExtractor(nn.Module):

    def __init__(self):

        super(FeatureExtractor, self).__init__()

        self.conv1 = nn.Conv2d(1 , 16 , 3 , 1 , "same")
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16 , 32 , 3 , 1 , "same")
        self.pool2 = nn.MaxPool2d(2, 2)      
    
        self.fc1 = nn.Linear(32 * 7 * 7 , 64)
        
    def forward(self , x):
        
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = torch.tanh(self.fc1(x))

        return x



class DistanceLayer(nn.Module):

    def __init__(self):

        super(DistanceLayer , self).__init__()

        self.register_buffer('epsilon', torch.Tensor([1e-8]))


    def forward(self, x1 , x2):
        
        sum_sqaure = torch.sum(torch.square(x1 - x2) , dim = 1 , keepdim = True)

        return torch.sqrt(torch.maximum(sum_sqaure , self.epsilon))
