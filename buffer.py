from common import *

class ReplayBuffer:
    """ """
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """ Implementaion of the Replay Buffer. Please refer to the dqn/solution directory inside here:
        https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn
        
        Params:
        ******
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.action_size = action_size
        self.seed = random.seed(seed)
    def add(self, state, action, reward, next_state, done):
        """ Add a new experience to the tuple """
        sample = self.experience(state, action, reward, next_state, done)
        self.memory.append(sample)
        
    def sample(self):
        """ Sample a batch of experiences from memory """
        experiences_vector = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences_vector if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences_vector if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences_vector if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences_vector if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences_vector if e is not None])).float().to(device)
        
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """ return the size of the buffer """
        return len(self.memory)
