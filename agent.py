from common import *
from buffer import ReplayBuffer
from modelsQ import Actor, Critic
from OUNoise import OUNoise


#Module Variables
#Replay Buffer
BUFFER_SIZE = int(1e5) # memory replay buffer size
BATCH_SIZE = 64 # batch size
#Q Network Hyper-parameters
GAMMA = 0.99 # Q learning discount size
TAU = 1e-3 # for soft update from local network to taget network
LR_ACTOR = 5e-4 # learning rate
LR_CRITIC = 5e-4 # learning rate
#Update frequencies
UPDATE_CRITIC_EVERY = 1 # number of frames used to update the local network
UPDATE_ACTOR_TARGET = 2 * UPDATE_CRITIC_EVERY
NN_NUM_UPDATES = 2

#Noise Parameters
NOISE_SCALE = 1.0

class AgentTD3():
    """ 
        Implementaion of TD3 Agent. 
        Please refer to the TD3 description and implementation here:
        https://spinningup.openai.com/en/latest/algorithms/td3.html
        https://github.com/sfujim/TD3
    """
    def __init__(self, state_size, action_size, num_agents, seed):
        """ Initialization of the agent class 
        
        Params
        ******
            state_size (int): Observation (state) vector size
            action_size (int): Action vector size
            seed (int): Random seed
        """
        #Environment Defined Variables
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(seed)

        #Actor Networks
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        #Critic Networks
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_ACTOR)
        
        #Buffer - Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        #Time step variable
        self.t_step = 0
        #OUNoise
        self.noise = OUNoise(action_size, scale=NOISE_SCALE)
        
    def step(self, states, actions, rewards, next_states, dones):
        """ Collection of experiences and learning from those experiences 
        Params
        ******
            state (Numpy Array): previous observed agent state
            action (int): Action taken to change from state to next_state
            reward (float): Reward acquired from state-action pair
            next_state (Numpy Array): Next observed agent state
            done (boolean): Sequence is done boolean
        """
        #Update memory
            #pdb.set_trace() - For Debugging
        self.t_step += 1
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
           #pdb.set_trace()
            self.memory.add(state, action, reward, next_state, done)
        #Learn every number of time steps according to variable UPDATE_EVERY
        if self.t_step % UPDATE_CRITIC_EVERY == 0:
            #Learn from enough experiences
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, states, n_factor=0.0, use_target=False, add_noise=False):
        """ 
        Compute the action based on the current state vector
        The noise is implemented based on Mr. Fujimoto's concept of noise
        Params
        ******
            state (Numpy Array): current state
            n_factor (Float): noise weight factor 
            use_target (Boolean): return target actions
            add_noise (Boolean): add noise to the actions 
        """
        actions = np.array([])
        #pdb.set_trace()
        for idx, state in enumerate(states):
            state = torch.from_numpy(state).float().to(device) if not torch.is_tensor(state) else state
            self.actor_local.eval()
            with torch.no_grad():
                if use_target:
                    action_values = self.actor_target(state).cpu().data.numpy().astype('float64')
                else:
                    action_values = self.actor_local(state).cpu().data.numpy().astype('float64')
                #pdb.set_trace()
            self.actor_local.train()
            #pdb.set_trace()
            # Add noise to the action vector
            if add_noise:
                #pdb.set_trace()
                noise = self.noise.noise().cpu().data.numpy().astype('float64')
                action_values = np.clip(action_values + n_factor*noise, -1, 1)
                #pdb.set_trace()
            else:
                action_values = np.clip(action_values, -1, 1)
            #pdb.set_trace()
            actions = np.append(actions, action_values)

        #pdb.set_trace()
        return np.reshape(actions,(states.shape[0], self.action_size))
        
    def learn(self, experiences, gamma):
        """ 
        
        Params
        ******
            experiences (Pytorch Tensor Tuple): Experience tuple - (state, action, reward, next_state, done)
            gamma (float): Q learning discount factor
        """
        for update_idx in range(1,NN_NUM_UPDATES+1):

            states, actions, rewards, next_states, dones = experiences
            #db.set_trace()
            # ------------------------------- Update Critic Network ---------------------------------------------- #
            #Adding noise to the target actions
            #pdb.set_trace()
            actions_next = self.act(next_states, n_factor=0.25, use_target=True, add_noise=True)
            actions_next = torch.from_numpy(actions_next).float().to(device)
            # Compute Target Q Values
            Q1_Target, Q2_Target = self.critic_target(next_states, actions_next)
            Q_Target_next = torch.min(Q1_Target, Q2_Target)
            Q_Targets = rewards + (gamma * Q_Target_next * (1 - dones))
            
            #Expected Q values from local network
            Q1_expected, Q2_expected = self.critic_local(states, actions)

            #Compute the loss
            loss = F.mse_loss(Q1_expected, Q_Targets) + F.mse_loss(Q2_expected, Q_Targets)
            #Minimize the loss
            #pdb.set_trace()
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

            # ------------------------------- Update Local Actor and Target Networks ------------------------------ #
            if update_idx % 2 == 0:
                #Update local actor network !
                actions_pred = self.actor_local(states)
                actor_loss = -self.critic_local.Q1(states, actions_pred).mean()
                #Minimize the loss
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                #Slow annealing of target network w.r.t local network
                self.soft_update(self.actor_local, self.actor_target, TAU)
                #Update Frozen Target Critic Network
                self.soft_update(self.critic_local, self.critic_target, TAU)
    
    def soft_update(self, local_model, target_model, tau):
        """ Soft annealing of the target network weights w.r.t to the local network weights
        
        Params
        ******
            local_model (Pytorch Network): Local Network
            target_model (Pytorch Network): Tarrget Network
            tau (float): Annealing factor 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-TAU)*target_param.data)
        
        
