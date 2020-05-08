#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[3]:


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_size = 100):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        
        self.state_size = state_size
        self.drop_prob = 0.1
        self.action_size = action_size
        self.hidden_size = hidden_size
        
        torch.manual_seed(seed)
        self.l1 = nn.Linear(in_features = self.state_size, out_features = hidden_size, bias = True)
        # self.batch1 = nn.BatchNorm1d(num_features = 50)
        self.l2 = nn.Linear(in_features = hidden_size, out_features = hidden_size/2, bias = True)
        # self.batch2 = nn.BatchNorm1d(num_features = 50)
        self.l3 = nn.Linear(in_features = hidden_size/2, out_features = hidden_size/4, bias = True)
        # self.batch3 = nn.BatchNorm1d(num_features = 50)
        self.l4 = nn.Linear(in_features = hidden_size/4, out_features = action_size, bias = True)
        self.dropout = nn.Dropout(p = self.drop_prob)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        x = F.selu(self.l1(state))
        # x = self.dropout(x)
        x = F.selu(self.l2(x))
        # x = self.dropout(x)
        x = F.selu(self.l3(x))
        # x = self.dropout(x)
        x = self.l4(x)
        
        return x 

