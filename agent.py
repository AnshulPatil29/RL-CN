# In agent.py, replace the entire AgentManager class with this corrected version.

import torch
import torch.optim as optim
from torch.distributions import Categorical
from models import PolicyGNN
import config

class AgentManager:
    def __init__(self, env):
        self.env = env
        self.num_agents = env.num_agents
        
        # Filter out agents with no links to control
        self.active_agents = [i for i, links in env.agent_link_partition.items() if links]
        self.num_active_agents = len(self.active_agents)
        
        self.policy = PolicyGNN(
            in_channels=2,
            hidden_channels=config.GNN_HIDDEN_DIM,
            num_layers=config.NUM_GNN_LAYERS,
            num_agents=self.num_agents,
            agent_partitions=env.agent_link_partition
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.LEARNING_RATE)
        
        # --- IMPROVEMENT: Buffers are now instance variables, managed by a helper ---
        self.clear_buffers()

    def clear_buffers(self):
        """Initializes or clears the experience buffers for a new rollout."""
        self.rewards = []
        # Buffers below are lists of lists, indexed by [agent_id][time_step]
        self.log_probs = [[] for _ in range(self.num_agents)]
        self.values = [[] for _ in range(self.num_agents)]
        self.entropies = [[] for _ in range(self.num_agents)]

    def select_actions(self, pyg_data):
        """Selects actions for all agents and stores values, log_probs, and entropies."""
        all_agent_logits, all_agent_values = self.policy(pyg_data)
        
        actions = {}
        
        # --- IMPROVEMENT: We now APPEND to buffers, not re-initialize them ---
        for i, agent_id in enumerate(self.active_agents):
            logits = all_agent_logits[i]
            value = all_agent_values[i]
            
            # Store the value for this agent at this timestep
            self.values[agent_id].append(value)

            num_links = len(self.env.agent_link_partition[agent_id])
            reshaped_logits = logits.view(num_links, config.ACTION_SPACE_SIZE)
            
            dist = Categorical(logits=reshaped_logits)
            action = dist.sample()
            
            actions[agent_id] = action.tolist()
            
            # Store the sum of log_probs and entropies for this agent's actions
            self.log_probs[agent_id].append(dist.log_prob(action).sum())
            self.entropies[agent_id].append(dist.entropy().sum())
            
        return actions

    def update(self, next_pyg_data):
        """Performs a full A2C update after a rollout of N_STEPS."""
        
        # 1. Bootstrap the value of the last state
        with torch.no_grad():
            _, next_values = self.policy(next_pyg_data)

        # 2. Calculate returns and advantages
        num_steps = len(self.rewards)
        returns = torch.zeros(self.num_active_agents, num_steps)
        advantages = torch.zeros(self.num_active_agents, num_steps)
        
        R = [v.item() for v in next_values]
        
        for t in reversed(range(num_steps)):
            reward_t = self.rewards[t]
            R = [reward_t + config.GAMMA * r_next for r_next in R]
            returns[:, t] = torch.tensor(R)
            
            # Get value estimates for timestep t for all active agents
            step_values = torch.tensor([self.values[agent_id][t].item() for agent_id in self.active_agents])
            
            advantages[:, t] = torch.tensor(R) - step_values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3. Calculate losses
        # Collate log_probs and entropies from the buffers
        log_probs_tensor = torch.stack([torch.stack(self.log_probs[i]) for i in self.active_agents])
        entropies_tensor = torch.stack([torch.stack(self.entropies[i]) for i in self.active_agents])
        
        # Align advantages shape (agents, steps) with log_probs (agents, steps)
        policy_loss = - (log_probs_tensor * advantages).mean()
        
        # Collate values and align with returns for MSE loss
        all_values_tensor = torch.stack([torch.cat(self.values[i]) for i in self.active_agents]).squeeze()
        value_loss = torch.nn.functional.mse_loss(all_values_tensor, returns)

        entropy_loss = -entropies_tensor.mean()
        
        loss = policy_loss + config.VALUE_LOSS_COEF * value_loss + config.ENTROPY_COEF * entropy_loss

        # 4. Backpropagate and update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
        
        # --- IMPROVEMENT: Clear buffers after the update is complete ---
        self.clear_buffers()