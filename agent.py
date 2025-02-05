import numpy as np
import random
import copy
from collections import namedtuple, deque
from itertools import count
import time

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

_batch_size = 256       # minibatch size
_buffer_size = int(1e5) # replay buffer size
_gamma = 0.99           # discount factor
_lr_actor = 1e-4        # learning rate of the actor 
_lr_critic = 1e-4       # learning rate of the critic
_tau = 3e-1             # soft update interpolation
_noise_decay = 0.999    # OU Noise decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed, num_agents):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            num_agents (int) : number of agents
        """
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=_lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=_lr_critic, weight_decay=0)

        # Noise process
        self.noise_decay = _noise_decay
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, _buffer_size, _batch_size, random_seed)
    
    def step(self, states, actions, rewards, next_states, dones):
        """[FOR EACH AGENT]Save experience in replay memory, and use random sample from buffer to learn."""
        for i in range(self.num_agents):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

        # Learn, if enough samples are available in memory
        if len(self.memory) > _batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, _gamma)

    def act(self, states, add_noise=True, apply_noise_decay=False):
        """Returns actions for given states as per current policy."""
        states = torch.from_numpy(states).float().to(device)
        actions = []
        self.actor_local.eval()
        with torch.no_grad():
            for state in states:
                # Take an action for each agent (for each state)
                action = self.actor_local(state).cpu().data.numpy()
                actions.append(action)
        self.actor_local.train()
        if add_noise:
            noise = self.noise.sample()
            if apply_noise_decay:
                actions += self.noise_decay*noise
                self.noise_decay *= self.noise_decay
            else:
                actions += noise
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, _tau)
        self.soft_update(self.actor_local, self.actor_target, _tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def train(self, env, n_episodes=3000, checkpoint_file='checkpoint.pt', print_every=10, apply_noise_decay=False):
        """Deep Deterministic Policy Gradients (DDPG).
        Params
        ======
            env: environment
            n_episodes (int): maximum number of training episodes
            checkpoint_file(str): model file
        """  
        
        scores_deque = deque(maxlen=100)
        scores_all = []
        moving_average = []
        
        # get the default brain
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]

        for i_episode in range(1, n_episodes+1):

            # Keep track of the current timestep
            timestep = time.time()

            ###################
            # FOR EACH AGENT  #
            ###################
            env_info = env.reset(train_mode=True)[brain_name]      # reset env    
            states = env_info.vector_observations                  # get current state
            scores = np.zeros(self.num_agents)                     # init score

            self.reset()                                           # reset the noise
 
            score_average = 0

            for t in count():
                actions = self.act(states, 
                                   apply_noise_decay=apply_noise_decay) # get actions (for each state==for each agent)
                env_info = env.step(actions)[brain_name]           # send actions to env
                next_states = env_info.vector_observations         # get next state
                rewards = env_info.rewards                         # get reward
                dones = env_info.local_done                        # get dones

                #Step agents (for each state==for each agent)
                self.step(states, actions, rewards, next_states, dones)

                #update states and scores:
                states = next_states                               # roll over states to next time step
                scores += rewards                                  # update the score (for each agent)            

                if np.any(dones):                                  # exit loop if episode finished
                    break

            score = np.max(scores)                                 # take max of agent scores
            scores_deque.append(score)
            score_average = np.mean(scores_deque)
            scores_all.append(score)
            moving_average.append(score_average)

            # Every n episodes print score
            if i_episode % print_every == 0:
                print('\rEpisode {}, Average Score: {:.2f}, Max: {:.2f}, Min: {:.2f}, Avg: {:.2f}, Time: {:.2f}'\
                  .format(i_episode, score_average, 
                          np.max(scores_all[-print_every:]), 
                          np.min(scores_all[-print_every:]), 
                          np.mean(scores_all[-print_every:]),
                          time.time() - timestep), end="\n")        

            # The environment is considered solved, 
            # when the scores average (over 100 episodes) is at least +0.5.
            if score_average >= 0.5:
                print('\n\nEnvironment solved in {:d} episodes!\t' \
                  'Moving Average Score: {:.3f}'
                  .format(i_episode, moving_average[-1]))
                self.save(checkpoint_file)                
                break            

        return scores_all, moving_average                    

    def save(self, file='checkpoint.pt'):
        """ Save the model """
        checkpoint = {
            'actor_dict': self.actor_local.state_dict(),
            'critic_dict': self.critic_local.state_dict()
            }
        print('\nSaving model ...', end=' ')
        torch.save(checkpoint, file)
        print('done.')
        
    def load(self, file='checkpoint.pt', map_location='cpu'):
        """ Load the trained model """
        checkpoint = torch.load(file, map_location=map_location)
        self.actor_local.load_state_dict(checkpoint['actor_dict'])
        self.critic_local.load_state_dict(checkpoint['critic_dict'])    
        
    def test(self, env, n_episodes=5):
        """Test the agend
        Params
        ======
            env: environment
            n_episodes (int): maximum number of testing episodes
        """
        for i in range(n_episodes):
            # get the default brain                                 
            brain_name = env.brain_names[0]
            brain = env.brains[brain_name]                           
            env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
            states = env_info.vector_observations                  # get the current state (for each agent)
            scores = np.zeros(self.num_agents)                     # initialize the score (for each agent)
            while True:
                actions = self.act(states, add_noise=False)        # select an action (for each agent)
                env_info = env.step(actions)[brain_name]           # send all actions to tne environment
                next_states = env_info.vector_observations         # get next state (for each agent)
                rewards = env_info.rewards                         # get reward (for each agent)
                dones = env_info.local_done                        # see if episode finished
                scores += env_info.rewards                         # update the score (for each agent)
                states = next_states                               # roll over states to next time step
                if np.any(dones):                                  # exit loop if episode finished
                    break                                          
            print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
                                          
class OUNoise():
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.gauss(0., 1.) for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer():
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)