from common import *

class Actor(nn.Module):
    """ 
        Implementing the Actor Neural Network 
        Please refer to Mr. Fujimoto's Implementation found here: https://github.com/sfujim/TD3    
    """
    
    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        """
        Neural net model parameters
        Params:
        ******
            state_size (int): Observation (state) vector size
            action_size (int): Action vector size
            seed (int): Random seed
            fc1_units (int): Number of nodes in first network layer
            fc1_unit (int): Number of nodes in second network layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
    def forward(self, state):
        """
        Forward pass of the state through the model to compute the corresponding action. 
        """
        #pdb.set_trace() - For debugging
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))
    
class Critic(nn.Module):
    """ 
        Implementing the Critic Neural Network
        Please refer to Mr. Fujimoto's Implementation found here: https://github.com/sfujim/TD3
    """
    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=256):
        """
        Neural net model parameters
        Params:
        ******
            state_size (int): Observation (state) vector size
            action_size (int): Action vector size
            seed (int): Random seed
            fc1_units (int): Number of nodes in first network layer
            fc1_unit (int): Number of nodes in second network layer
        """
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_size + action_size, fc1_units)
        #Q1 Implementation
        self.fc2_q1 = nn.Linear(fc1_units, fc2_units)
        self.fc3_q1 = nn.Linear(fc2_units, 1)
        #Q2 Implementation
        self.fc2_q2 = nn.Linear(fc1_units, fc2_units)
        self.fc3_q2 = nn.Linear(fc2_units, 1)

    def forward(self, state, action):
        """
        Forward pass of the state through the model to compute the corresponding action = advantage + value. 
        """
        Q1 = self.Q1(state, action)
        Q2 = self.Q2(state, action)

        return Q1, Q2
    
    def Q1(self, state, action):
        """
        Forward pass definition of the Q1 Network
        """
        xsa = torch.cat((state, action), dim=1)
        #pdb.set_trace()
        x = F.relu(self.fc1(xsa))
        
        x = F.relu(self.fc2_q1(x))
        return self.fc3_q1(x)
    
    def Q2(self, state, action):
        """
        Forward pass definition of the Q2 Network
        """
        xsa = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(xsa))
        
        x = F.relu(self.fc2_q2(x))
        return self.fc3_q2(x)

