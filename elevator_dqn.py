# elevator_dqn.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def put(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))
    
    def get(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones, dtype=bool)
        )
    
    def __len__(self):
        return len(self.buffer)

class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# class DQNNetwork(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(DQNNetwork, self).__init__()
#         # Larger network to handle complex state representation
#         self.fc1 = nn.Linear(input_dim, 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, 128)
#         self.fc4 = nn.Linear(128, 64)
#         self.fc5 = nn.Linear(64, output_dim)
        
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = torch.relu(self.fc3(x))
#         x = torch.relu(self.fc4(x))
#         return self.fc5(x)

class ElevatorDQN:
    def __init__(self, env, replay_size=50000, batch_size=64, gamma=0.99, 
                 sync_after=1000, lr=0.0001, resume=False, device=None):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        
        # Handle different action spaces from your environment
        if hasattr(env.action_space, 'n'):
            self.act_dim = env.action_space.n
        elif hasattr(env.action_space, 'shape'):
            # For continuous or multi-discrete, we need to adapt
            self.act_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.nvec.prod()
        else:
            raise ValueError(f"Unsupported action space: {env.action_space}")
            
        self.replay_buffer = ReplayBuffer(replay_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.sync_after = sync_after
        self.training_step = 0
        
        # Device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network
        self.dqn_net = DQNNetwork(self.obs_dim, self.act_dim).to(self.device)
        self.dqn_target_net = DQNNetwork(self.obs_dim, self.act_dim).to(self.device)
        
        if resume:
            try:
                self.dqn_net.load_state_dict(torch.load('./models/elevator_dqn.pth', map_location=self.device))
                print("Loaded existing DQN model")
            except FileNotFoundError:
                print("No existing DQN model found, starting fresh")
        
        self.dqn_target_net.load_state_dict(self.dqn_net.state_dict())
        self.dqn_target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.dqn_net.parameters(), lr=lr, weight_decay=1e-5)
        
        # Tracking
        self.losses = []
        self.all_rewards = []
    
    def learn(self, timesteps, save_path='./models/elevator_dqn.pth'):
        """Train the DQN agent with your advanced environment"""
        all_rewards = []
        episode_rewards = []
        
        obs, info = self.env.reset()
        
        for timestep in range(1, timesteps + 1):
            epsilon = self.epsilon_by_timestep(timestep)
            action = self.predict(obs, epsilon)
            
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            self.replay_buffer.put(obs, action, reward, next_obs, terminated)
            
            obs = next_obs
            episode_rewards.append(reward)
            
            if terminated or truncated:
                total_episode_reward = sum(episode_rewards)
                all_rewards.append(total_episode_reward)
                obs, info = self.env.reset()
                episode_rewards = []
                
                if len(all_rewards) % 10 == 0:
                    avg_reward = np.mean(all_rewards[-5:])
                    completed_passengers = info.get("passengers_completed", 0)
                    avg_wait = info.get("average_wait_time", 0)
                    print(f"DQN - Episode {len(all_rewards)}, "
                          f"Avg Reward: {avg_reward:.2f}, "
                          f"Completed: {completed_passengers}, "
                          f"Avg Wait: {avg_wait:.1f}s, "
                          f"Epsilon: {epsilon:.3f}")
            
            # Training
            if len(self.replay_buffer) > self.batch_size:
                loss = self.compute_msbe_loss()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.dqn_net.parameters(), 1.0)
                self.optimizer.step()
                self.training_step += 1
                self.losses.append(loss.item())
            
            # Sync target network
            if self.training_step > 0 and self.training_step % self.sync_after == 0:
                self.dqn_target_net.load_state_dict(self.dqn_net.state_dict())
            
            # Save model periodically
            if timestep % 5000 == 0:
                torch.save(self.dqn_net.state_dict(), save_path)
                print(f"Saved model at step {timestep}")
        
        # Final save
        torch.save(self.dqn_net.state_dict(), save_path)
        print("DQN Training completed!")
        
        return all_rewards
    
    def predict(self, state, epsilon=0.0):
        """Predict action based on state - handles different action spaces"""
        if random.random() < epsilon:
            # Random action based on action space type
            if hasattr(self.env.action_space, 'n'):
                return random.randint(0, self.act_dim - 1)
            elif hasattr(self.env.action_space, 'nvec'):
                return np.array([random.randint(0, n-1) for n in self.env.action_space.nvec])
            else:
                return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.dqn_net(state_tensor)
                
                if hasattr(self.env.action_space, 'n'):
                    return q_values.argmax().item()
                else:
                    # For multi-discrete, return the action with highest Q-value
                    return q_values.cpu().numpy().flatten()
    
    def compute_msbe_loss(self):
        """Compute MSBE loss"""
        obs, actions, rewards, next_obs, dones = self.replay_buffer.get(self.batch_size)
        
        # Convert to tensors
        obs = torch.FloatTensor(obs).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.dqn_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q values
        with torch.no_grad():
            next_q_values = self.dqn_target_net(next_obs).max(1)[0]
            next_q_values[dones] = 0.0
            target_q_values = rewards + self.gamma * next_q_values
        
        # Compute loss
        loss = torch.nn.MSELoss()(current_q_values, target_q_values)
        return loss
    
    def epsilon_by_timestep(self, timestep, epsilon_start=1.0, epsilon_final=0.01, frames_decay=50000):
        """Epsilon decay for elevator problem"""
        if timestep >= frames_decay:
            return epsilon_final
        else:
            return epsilon_start - (epsilon_start - epsilon_final) * (timestep / frames_decay)

class ElevatorDDQN:
    def __init__(self, env, replay_size=50000, batch_size=64, gamma=0.99, 
                 sync_after=1000, lr=0.0001, resume=False, device=None):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        
        # Handle different action spaces
        if hasattr(env.action_space, 'n'):
            self.act_dim = env.action_space.n
        else:
            self.act_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.nvec.prod()
            
        self.replay_buffer = ReplayBuffer(replay_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.sync_after = sync_after
        self.training_step = 0
        
        # Device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks for DDQN
        self.dqn_net = DQNNetwork(self.obs_dim, self.act_dim).to(self.device)
        self.dqn_target_net = DQNNetwork(self.obs_dim, self.act_dim).to(self.device)
        
        if resume:
            try:
                self.dqn_net.load_state_dict(torch.load('./models/elevator_ddqn.pth', map_location=self.device))
                print("Loaded existing DDQN model")
            except FileNotFoundError:
                print("No existing DDQN model found, starting fresh")
        
        self.dqn_target_net.load_state_dict(self.dqn_net.state_dict())
        self.dqn_target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.dqn_net.parameters(), lr=lr, weight_decay=1e-5)
        
        # Tracking
        self.losses = []
        self.all_rewards = []
    
    def learn(self, timesteps, save_path='elevator_ddqn.pth'):
        """Train the DDQN agent with your advanced environment"""
        all_rewards = []
        episode_rewards = []
        
        obs, info = self.env.reset()
        
        for timestep in range(1, timesteps + 1):
            epsilon = self.epsilon_by_timestep(timestep)
            action = self.predict(obs, epsilon)
            
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            self.replay_buffer.put(obs, action, reward, next_obs, terminated)
            
            obs = next_obs
            episode_rewards.append(reward)
            
            if terminated or truncated:
                total_episode_reward = sum(episode_rewards)
                all_rewards.append(total_episode_reward)
                obs, info = self.env.reset()
                episode_rewards = []
                
                if len(all_rewards) % 10 == 0:
                    avg_reward = np.mean(all_rewards[-5:])
                    completed_passengers = info.get("passengers_completed", 0)
                    avg_wait = info.get("average_wait_time", 0)
                    print(f"DDQN - Episode {len(all_rewards)}, "
                          f"Avg Reward: {avg_reward:.2f}, "
                          f"Completed: {completed_passengers}, "
                          f"Avg Wait: {avg_wait:.1f}s, "
                          f"Epsilon: {epsilon:.3f}")
            
            # Training
            if len(self.replay_buffer) > self.batch_size:
                loss = self.compute_ddqn_loss()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.dqn_net.parameters(), 1.0)
                self.optimizer.step()
                self.training_step += 1
                self.losses.append(loss.item())
            
            # Sync target network
            if self.training_step > 0 and self.training_step % self.sync_after == 0:
                self.dqn_target_net.load_state_dict(self.dqn_net.state_dict())
            
            # Save model periodically
            if timestep % 5000 == 0:
                torch.save(self.dqn_net.state_dict(), save_path)
                print(f"Saved DDQN model at step {timestep}")
        
        # Final save
        torch.save(self.dqn_net.state_dict(), save_path)
        print("DDQN Training completed!")
        
        return all_rewards
    
    def predict(self, state, epsilon=0.0):
        """Predict action based on state"""
        if random.random() < epsilon:
            if hasattr(self.env.action_space, 'n'):
                return random.randint(0, self.act_dim - 1)
            else:
                return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.dqn_net(state_tensor)
                
                if hasattr(self.env.action_space, 'n'):
                    return q_values.argmax().item()
                else:
                    return q_values.cpu().numpy().flatten()
    
    def compute_ddqn_loss(self):
        """Compute Double DQN loss"""
        obs, actions, rewards, next_obs, dones = self.replay_buffer.get(self.batch_size)
        
        # Convert to tensors
        obs = torch.FloatTensor(obs).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.dqn_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: Action selection by online network, evaluation by target network
        with torch.no_grad():
            next_actions = self.dqn_net(next_obs).argmax(1)
            next_q_values = self.dqn_target_net(next_obs).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            next_q_values[dones] = 0.0
            target_q_values = rewards + self.gamma * next_q_values
        
        # Compute loss
        loss = torch.nn.MSELoss()(current_q_values, target_q_values)
        return loss
    
    def epsilon_by_timestep(self, timestep, epsilon_start=1.0, epsilon_final=0.01, frames_decay=50000):
        """Epsilon decay"""
        if timestep >= frames_decay:
            return epsilon_final
        else:
            return epsilon_start - (epsilon_start - epsilon_final) * (timestep / frames_decay)

class ElevatorTDQN:
    def __init__(self, env, replay_size=50000, batch_size=64, gamma=0.99, 
                 sync_after1=1000, sync_after2=5000, lr=0.0001, aggregator='min', 
                 resume=False, device=None):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        
        # Handle different action spaces
        if hasattr(env.action_space, 'n'):
            self.act_dim = env.action_space.n
        else:
            self.act_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.nvec.prod()
            
        self.replay_buffer = ReplayBuffer(replay_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.sync_after1 = sync_after1  # Sync frequency for target1
        self.sync_after2 = sync_after2  # Sync frequency for target2
        self.aggregator = aggregator
        self.training_step = 0
        
        # Device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Three networks for TDQN
        self.dqn_net = DQNNetwork(self.obs_dim, self.act_dim).to(self.device)
        self.dqn_target1 = DQNNetwork(self.obs_dim, self.act_dim).to(self.device)
        self.dqn_target2 = DQNNetwork(self.obs_dim, self.act_dim).to(self.device)
        
        if resume:
            try:
                checkpoint = torch.load('./models/elevator_tdqn.pth', map_location=self.device)
                self.dqn_net.load_state_dict(checkpoint['online'])
                self.dqn_target1.load_state_dict(checkpoint['target1'])
                self.dqn_target2.load_state_dict(checkpoint['target2'])
                print("Loaded existing TDQN model")
            except FileNotFoundError:
                print("No existing TDQN model found, starting fresh")
        
        # Initialize targets
        self.dqn_target1.load_state_dict(self.dqn_net.state_dict())
        self.dqn_target2.load_state_dict(self.dqn_net.state_dict())
        self.dqn_target1.eval()
        self.dqn_target2.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.dqn_net.parameters(), lr=lr, weight_decay=1e-5)
        
        # Tracking
        self.losses = []
        self.all_rewards = []
    
    def learn(self, timesteps, save_path='./models/elevator_tdqn.pth'):
        """Train the TDQN agent with your advanced environment"""
        all_rewards = []
        episode_rewards = []
        
        obs, info = self.env.reset()
        
        for timestep in range(1, timesteps + 1):
            epsilon = self.epsilon_by_timestep(timestep)
            action = self.predict(obs, epsilon)
            
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            self.replay_buffer.put(obs, action, reward, next_obs, terminated)
            
            obs = next_obs
            episode_rewards.append(reward)
            
            if terminated or truncated:
                total_episode_reward = sum(episode_rewards)
                all_rewards.append(total_episode_reward)
                obs, info = self.env.reset()
                episode_rewards = []
                
                if len(all_rewards) % 10 == 0:
                    avg_reward = np.mean(all_rewards[-5:])
                    completed_passengers = info.get("passengers_completed", 0)
                    avg_wait = info.get("average_wait_time", 0)
                    print(f"TDQN - Episode {len(all_rewards)}, "
                          f"Avg Reward: {avg_reward:.2f}, "
                          f"Completed: {completed_passengers}, "
                          f"Avg Wait: {avg_wait:.1f}s, "
                          f"Epsilon: {epsilon:.3f}")
            
            # Training
            if len(self.replay_buffer) > self.batch_size:
                loss = self.compute_tdqn_loss()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.dqn_net.parameters(), 1.0)
                self.optimizer.step()
                self.training_step += 1
                self.losses.append(loss.item())
            
            # Staggered target network synchronization
            if self.training_step > 0 and self.training_step % self.sync_after1 == 0:
                self.dqn_target1.load_state_dict(self.dqn_net.state_dict())
            
            if self.training_step > 0 and self.training_step % self.sync_after2 == 0:
                self.dqn_target2.load_state_dict(self.dqn_net.state_dict())
            
            # Save model periodically
            if timestep % 5000 == 0:
                checkpoint = {
                    'online': self.dqn_net.state_dict(),
                    'target1': self.dqn_target1.state_dict(),
                    'target2': self.dqn_target2.state_dict()
                }
                torch.save(checkpoint, save_path)
                print(f"Saved TDQN model at step {timestep}")
        
        # Final save
        checkpoint = {
            'online': self.dqn_net.state_dict(),
            'target1': self.dqn_target1.state_dict(),
            'target2': self.dqn_target2.state_dict()
        }
        torch.save(checkpoint, save_path)
        print("TDQN Training completed!")
        
        return all_rewards
    
    def predict(self, state, epsilon=0.0):
        """Predict action based on state"""
        if random.random() < epsilon:
            if hasattr(self.env.action_space, 'n'):
                return random.randint(0, self.act_dim - 1)
            else:
                return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.dqn_net(state_tensor)
                
                if hasattr(self.env.action_space, 'n'):
                    return q_values.argmax().item()
                else:
                    return q_values.cpu().numpy().flatten()
    
    def compute_tdqn_loss(self):
        """Compute Triple DQN loss"""
        obs, actions, rewards, next_obs, dones = self.replay_buffer.get(self.batch_size)
        
        # Convert to tensors
        obs = torch.FloatTensor(obs).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.dqn_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Triple DQN
        with torch.no_grad():
            next_actions = self.dqn_net(next_obs).argmax(1)
            
            # Q-value evaluation by two separate target networks
            q_target1 = self.dqn_target1(next_obs).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            q_target2 = self.dqn_target2(next_obs).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # Aggregate the two target Q-values
            if self.aggregator == 'min':
                next_q_values = torch.min(q_target1, q_target2)
            else:  # mean
                next_q_values = 0.5 * (q_target1 + q_target2)
            
            next_q_values[dones] = 0.0
            target_q_values = rewards + self.gamma * next_q_values
        
        # Compute loss
        loss = torch.nn.MSELoss()(current_q_values, target_q_values)
        return loss
    
    def epsilon_by_timestep(self, timestep, epsilon_start=1.0, epsilon_final=0.01, frames_decay=50000):
        """Epsilon decay"""
        if timestep >= frames_decay:
            return epsilon_final
        else:
            return epsilon_start - (epsilon_start - epsilon_final) * (timestep / frames_decay)
