# mappo_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# -----------------------------
# Hyperparameters (tweak as needed)
# -----------------------------
HIDDEN = 64
GAMMA = 0.99
LR = 1e-3
CLIP = 0.2
UPDATE_EPOCHS = 4
BATCH_SIZE = 32
EPS = 1e-8
DEVICE = torch.device("cpu")  # laptop-friendly

# -----------------------------
# Actor Network (per agent)
# -----------------------------
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, action_dim)
        )
    
    def forward(self, x):
        logits = self.fc(x)
        return torch.softmax(logits, dim=-1)

# -----------------------------
# Critic Network (centralized)
# -----------------------------
class Critic(nn.Module):
    def __init__(self, obs_dim_total):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim_total, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 1)
        )
    def forward(self, x):
        return self.fc(x)

# -----------------------------
# MAPPO Agent
# -----------------------------
class MAPPOAgent:
    def __init__(self, num_agents=4, obs_dim=16, action_dim=3):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Each agent has its own actor
        self.actors = [Actor(obs_dim, action_dim).to(DEVICE) for _ in range(num_agents)]
        self.actor_optim = [optim.Adam(actor.parameters(), lr=LR) for actor in self.actors]

        # Centralized critic sees global state
        self.critic = Critic(obs_dim * num_agents).to(DEVICE)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=LR)

        # Trajectory storage
        self.memory = []

    # -----------------------------
    # Convert observation dict to flat vector
    # -----------------------------
    @staticmethod
    def flatten_obs(obs):
        lights = obs['lights'].astype(np.float32) / 2.0  # normalize 0-1
        queues = obs['queues'].astype(np.float32) / 100.0 
        emergency = obs['emergency'].astype(np.float32)
        time = np.array([obs['time']/1000.0], dtype=np.float32)
        return np.concatenate([lights, queues, emergency, time])

    # -----------------------------
    # Select actions for all agents
    # -----------------------------
    def select_actions(self, obs):
        obs_vec = self.flatten_obs(obs)
        obs_tensor = torch.tensor(obs_vec, dtype=torch.float32).to(DEVICE)
        actions = []
        log_probs = []

        for actor in self.actors:
            probs = actor(obs_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            actions.append(action.item())
            log_probs.append(dist.log_prob(action))
        return actions, log_probs, obs_vec

    # -----------------------------
    # Store step in memory
    # -----------------------------
    def store_transition(self, obs_vec, actions, log_probs, reward, next_obs_vec, done):
        self.memory.append({
            "obs": obs_vec,
            "actions": actions,
            "log_probs": log_probs,
            "reward": reward,
            "next_obs": next_obs_vec,
            "done": done
        })

    # -----------------------------
    # PPO Update
    # -----------------------------
    def update(self):
        if not self.memory:
            return

        # Prepare batches
        obs_batch = torch.tensor([m["obs"] for m in self.memory], dtype=torch.float32).to(DEVICE)
        next_obs_batch = torch.tensor([m["next_obs"] for m in self.memory], dtype=torch.float32).to(DEVICE)
        rewards = torch.tensor([m["reward"] for m in self.memory], dtype=torch.float32).to(DEVICE)
        dones = torch.tensor([m["done"] for m in self.memory], dtype=torch.float32).to(DEVICE)

        # Centralized critic: estimate values
        values = self.critic(obs_batch.repeat(1, self.num_agents))
        next_values = self.critic(next_obs_batch.repeat(1, self.num_agents))
        advantages = rewards.unsqueeze(-1) + GAMMA * next_values * (1 - dones.unsqueeze(-1)) - values

        # Update actors
        for agent_idx, actor in enumerate(self.actors):
            optimizer = self.actor_optim[agent_idx]
            for _ in range(UPDATE_EPOCHS):
                old_log_probs = torch.stack([m["log_probs"][agent_idx] for m in self.memory])
                # Recompute log_probs
                probs = actor(obs_batch)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(torch.tensor([m["actions"][agent_idx] for m in self.memory], dtype=torch.long))
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages.detach()
                surr2 = torch.clamp(ratio, 1.0 - CLIP, 1.0 + CLIP) * advantages.detach()
                loss = -torch.min(surr1, surr2).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Update centralized critic
        for _ in range(UPDATE_EPOCHS):
            values_pred = self.critic(obs_batch.repeat(1, self.num_agents))
            critic_loss = (values_pred - (rewards.unsqueeze(-1) + GAMMA * next_values.detach() * (1 - dones.unsqueeze(-1))))**2
            critic_loss = critic_loss.mean()
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

        # Clear memory
        self.memory = []
