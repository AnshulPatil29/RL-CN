import os
import textwrap

# --- Project Configuration ---
ROOT_DIR = "marl-routing"

# --- File Contents as Multiline Strings ---

# Using textwrap.dedent to allow for clean indentation in the script
# without adding it to the output files.

CONFIG_PY_CONTENT = textwrap.dedent("""
    # --- Simulation Parameters ---
    TOPOLOGY_FILE = 'topologies/nsfnet.json'
    NUM_AGENTS = 4  # Number of areas/agents to partition the network into
    TRAFFIC_MODEL = 'gravity' # 'gravity' or 'random'
    MAX_STEPS_PER_EPISODE = 100 # Number of traffic matrices per episode

    # --- Training Parameters ---
    NUM_EPISODES = 500
    LEARNING_RATE = 1e-4
    GAMMA = 0.99  # Discount factor for future rewards

    # --- Agent and Model Parameters ---
    GNN_HIDDEN_DIM = 64
    NUM_GNN_LAYERS = 3
    ACTION_SPACE_SIZE = 10 # Number of discrete weights an agent can assign to a link (e.g., weights 1-10)

    # --- Reward Function Weights (Multi-Objective) ---
    # We want to minimize utilization and latency, so rewards are negative
    REWARD_WEIGHTS = {
        'utilization': -1.0,  # Penalty for high max link utilization
        'latency': -0.1       # Penalty for high average path latency
    }

    # --- Logging and Saving ---
    LOG_INTERVAL = 10
    MODEL_SAVE_PATH = 'results/marl_agent.pth'
    PLOT_SAVE_PATH = 'results/training_rewards.png'
""")

NETWORK_ENV_PY_CONTENT = textwrap.dedent("""
    import networkx as nx
    import numpy as np
    import json
    from networkx.readwrite import json_graph

    class NetworkEnv:
        def __init__(self, topology_file, num_agents):
            self.topology_file = topology_file
            self.num_agents = num_agents
            self.graph = self._load_topology()
            self.num_nodes = self.graph.number_of_nodes()
            # Count edges, ensuring not to double-count bidirectional links
            self.num_links = len(set(tuple(sorted(e)) for e in self.graph.edges()))
            self.agent_link_partition = self._partition_links()
            self.traffic_matrix = None

        def _load_topology(self):
            with open(self.topology_file) as f:
                data = json.load(f)
            
            G = json_graph.node_link_graph(data, directed=False)
            
            # Ensure graph is treated as undirected for our purposes
            G_undirected = nx.Graph()
            for u, v, attr in G.edges(data=True):
                G_undirected.add_edge(u, v, **attr)
            
            nx.set_edge_attributes(G_undirected, 1.0, 'weight') # OSPF-like cost
            nx.set_edge_attributes(G_undirected, 0.0, 'utilization')
            return G_undirected
        
        def _partition_links(self):
            # Simple partitioning: divide links among agents
            partition = {i: [] for i in range(self.num_agents)}
            links = list(self.graph.edges())
            
            # Create a sorted, unique list of edges to ensure consistent partitioning
            unique_links = sorted([tuple(sorted(e)) for e in links])
            unique_links = list(dict.fromkeys(unique_links))

            for idx, (u, v) in enumerate(unique_links):
                agent_id = idx % self.num_agents
                partition[agent_id].append((u, v))
            return partition

        def _generate_traffic_matrix(self, model='gravity'):
            # Generates a new traffic matrix for one step
            self.traffic_matrix = {}
            nodes = list(self.graph.nodes())
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if i != j:
                        if model == 'gravity':
                            # Simple gravity model placeholder
                            demand = np.random.uniform(5, 20)
                        else: # Random model
                            demand = np.random.uniform(1, 50) if np.random.rand() > 0.5 else 0
                        if demand > 0:
                            self.traffic_matrix[(i, j)] = demand
        
        def reset(self):
            # Reset environment for a new episode
            nx.set_edge_attributes(self.graph, 0.0, 'utilization')
            self._generate_traffic_matrix()
            return self._get_state()

        def step(self, actions):
            # 1. Apply actions (update link weights)
            for agent_id, agent_actions in actions.items():
                for i, (u, v) in enumerate(self.agent_link_partition[agent_id]):
                    # Action is an index from 0 to ACTION_SPACE_SIZE-1. Convert to weight.
                    weight = agent_actions[i] + 1 
                    self.graph[u][v]['weight'] = weight

            # 2. Route traffic and update utilization
            nx.set_edge_attributes(self.graph, 0.0, 'utilization')
            total_latency = 0
            total_traffic = 0

            for (src, dst), demand in self.traffic_matrix.items():
                try:
                    path = nx.shortest_path(self.graph, source=src, target=dst, weight='weight')
                    path_latency = len(path) - 1
                    total_latency += path_latency * demand
                    total_traffic += demand
                    
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        self.graph[u][v]['utilization'] += demand
                except nx.NetworkXNoPath:
                    continue

            # 3. Calculate state, reward, and done flag
            # Generate new traffic for the next step before getting state
            self._generate_traffic_matrix()
            next_state = self._get_state()
            reward = self._get_reward(total_latency, total_traffic)
            
            return next_state, reward, False, {}

        def _get_state(self):
            # Create a state representation for the GNN
            # Node features are based on incident link utilizations
            node_features = np.zeros((self.num_nodes, 2)) # [avg_norm_util, degree]
            
            for u, v, attr in self.graph.edges(data=True):
                util = attr['utilization']
                cap = attr['capacity']
                norm_util = min(util / cap, 1.0) # Normalize and cap at 1.0
                node_features[u][0] += norm_util
                node_features[v][0] += norm_util
            
            # Normalize by degree and add degree as a feature
            for i in range(self.num_nodes):
                degree = self.graph.degree[i]
                if degree > 0:
                    node_features[i][0] /= degree
                node_features[i][1] = degree
                
            return node_features

        def _get_reward(self, total_latency, total_traffic):
            from config import REWARD_WEIGHTS
            
            utilizations = [attr['utilization'] / attr['capacity'] for _, _, attr in self.graph.edges(data=True)]
            max_utilization = max(utilizations) if utilizations else 0
            
            avg_latency = total_latency / total_traffic if total_traffic > 0 else 0

            reward_util = REWARD_WEIGHTS['utilization'] * max_utilization
            reward_latency = REWARD_WEIGHTS['latency'] * avg_latency
            
            total_reward = reward_util + reward_latency
            return total_reward
""")

MODELS_PY_CONTENT = textwrap.dedent("""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data
    from torch_geometric.utils import from_networkx

    class ActorCriticGNN(nn.Module):
        def __init__(self, in_channels, hidden_channels, num_layers, action_dim):
            super(ActorCriticGNN, self).__init__()
            
            self.convs = nn.ModuleList()
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
                
            self.actor_head = nn.Linear(hidden_channels, action_dim)
            self.critic_head = nn.Linear(hidden_channels, 1)

        def forward(self, x, edge_index):
            for conv in self.convs:
                x = conv(x, edge_index)
                x = F.relu(x)
            
            action_logits = self.actor_head(x)
            state_values = self.critic_head(x)
            
            return action_logits, state_values

        @staticmethod
        def to_pyg_data(graph, node_features):
            # Convert networkx graph to PyG Data object
            pyg_graph = from_networkx(graph)
            edge_index = pyg_graph.edge_index
            x = torch.tensor(node_features, dtype=torch.float)
            return Data(x=x, edge_index=edge_index)
""")

AGENT_PY_CONTENT = textwrap.dedent("""
    import torch
    import torch.optim as optim
    from torch.distributions import Categorical
    from models import ActorCriticGNN

    class DRLAgent:
        def __init__(self, agent_id, links, node_feature_dim, action_dim, lr, gamma):
            self.id = agent_id
            self.links = links
            self.num_links = len(links)
            self.gamma = gamma
            
            self.model = ActorCriticGNN(
                in_channels=node_feature_dim, 
                hidden_channels=64, 
                num_layers=3, 
                action_dim=action_dim
            )
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
            
            self.saved_log_probs = []
            self.rewards = []
            self.state_values = []

        def select_actions(self, pyg_data):
            all_action_logits, all_state_values = self.model(pyg_data.x, pyg_data.edge_index)

            agent_actions = []
            agent_log_probs = []
            
            for u, v in self.links:
                # Combine node embeddings to get link-specific logits
                link_embedding_logits = (all_action_logits[u] + all_action_logits[v]) / 2.0
                
                m = Categorical(logits=link_embedding_logits)
                action = m.sample()
                
                agent_actions.append(action.item())
                agent_log_probs.append(m.log_prob(action))

            # Calculate a single state value for the agent's controlled area
            nodes_in_control = set([n for link in self.links for n in link])
            value_sum = 0
            for node_idx in nodes_in_control:
                value_sum += all_state_values[node_idx]
            
            avg_value = value_sum / len(nodes_in_control) if nodes_in_control else torch.tensor([0.0])
            
            self.saved_log_probs.append(torch.stack(agent_log_probs).mean()) # Store mean log_prob for the step
            self.state_values.append(avg_value)
            
            return agent_actions

        def update(self):
            R = 0
            returns = []

            for r in reversed(self.rewards):
                R = r + self.gamma * R
                returns.insert(0, R)
                
            returns = torch.tensor(returns, dtype=torch.float32)
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)

            log_probs = torch.stack(self.saved_log_probs)
            values = torch.cat(self.state_values).squeeze()

            advantage = returns - values.detach()
            
            policy_loss = (-log_probs * advantage).mean()
            value_loss = torch.nn.functional.mse_loss(values, returns)
            
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.rewards = []
            self.saved_log_probs = []
            self.state_values = []
""")

TRAIN_PY_CONTENT = textwrap.dedent("""
    import torch
    import numpy as np
    from network_env import NetworkEnv
    from agent import DRLAgent
    from models import ActorCriticGNN
    import config

    def main():
        print("--- Initializing Environment and Agents ---")
        env = NetworkEnv(config.TOPOLOGY_FILE, config.NUM_AGENTS)

        agents = {}
        for agent_id, links in env.agent_link_partition.items():
            if not links: continue
            agents[agent_id] = DRLAgent(
                agent_id=agent_id,
                links=links,
                node_feature_dim=2, # From network_env: [avg_norm_util, degree]
                action_dim=config.ACTION_SPACE_SIZE,
                lr=config.LEARNING_RATE,
                gamma=config.GAMMA
            )
        
        if not agents:
            print("Error: No agents were created. Check topology and NUM_AGENTS.")
            return

        episode_rewards = []
        print(f"--- Starting Training for {config.NUM_EPISODES} Episodes ---")

        for i_episode in range(config.NUM_EPISODES):
            state = env.reset()
            episode_reward_sum = 0
            
            for t in range(config.MAX_STEPS_PER_EPISODE):
                pyg_data = ActorCriticGNN.to_pyg_data(env.graph, state)
                
                actions = {}
                for agent_id, agent in agents.items():
                    actions[agent_id] = agent.select_actions(pyg_data)

                state, reward, done, _ = env.step(actions)
                
                for agent in agents.values():
                    agent.rewards.append(reward)

                episode_reward_sum += reward

            # Update agents at the end of the episode
            for agent in agents.values():
                agent.update()

            episode_rewards.append(episode_reward_sum / config.MAX_STEPS_PER_EPISODE)

            if (i_episode + 1) % config.LOG_INTERVAL == 0:
                avg_reward = np.mean(episode_rewards[-config.LOG_INTERVAL:])
                print(f'Episode {i_episode+1}/{config.NUM_EPISODES}\\tAverage reward: {avg_reward:.3f}')

        print("--- Training Complete ---")
        
        # Save a model
        torch.save(list(agents.values())[0].model.state_dict(), config.MODEL_SAVE_PATH)
        print(f"Model saved to {config.MODEL_SAVE_PATH}")

        # Plot results
        from utils import plot_rewards
        plot_rewards(episode_rewards, save_path=config.PLOT_SAVE_PATH)
        print(f"Training plot saved to {config.PLOT_SAVE_PATH}")

    if __name__ == '__main__':
        main()
""")

OSPF_BASELINE_PY_CONTENT = textwrap.dedent("""
    import networkx as nx
    from network_env import NetworkEnv
    import config

    def evaluate_ospf(env, num_steps=100):
        total_max_util = 0
        total_avg_latency = 0
        
        print("\\n--- Evaluating OSPF Baseline ---")
        
        nx.set_edge_attributes(env.graph, 1, 'weight')
        
        for i in range(num_steps):
            env._generate_traffic_matrix()
            
            nx.set_edge_attributes(env.graph, 0.0, 'utilization')
            total_latency_step = 0
            total_traffic_step = 0

            for (src, dst), demand in env.traffic_matrix.items():
                try:
                    path = nx.shortest_path(env.graph, source=src, target=dst, weight='weight')
                    path_latency = len(path) - 1
                    total_latency_step += path_latency * demand
                    total_traffic_step += demand
                    
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        env.graph[u][v]['utilization'] += demand
                except nx.NetworkXNoPath:
                    continue
                    
            utilizations = [attr['utilization'] / attr['capacity'] for _, _, attr in env.graph.edges(data=True)]
            total_max_util += max(utilizations) if utilizations else 0
            total_avg_latency += total_latency_step / total_traffic_step if total_traffic_step > 0 else 0

        avg_max_util = total_max_util / num_steps
        avg_latency = total_avg_latency / num_steps
        
        print(f"OSPF Avg Max Link Utilization: {avg_max_util:.4f}")
        print(f"OSPF Avg Path Latency: {avg_latency:.4f}")
        
        return avg_max_util, avg_latency

    if __name__ == '__main__':
        env = NetworkEnv(config.TOPOLOGY_FILE, num_agents=1)
        evaluate_ospf(env, num_steps=config.MAX_STEPS_PER_EPISODE * config.NUM_EPISODES)
""")

UTILS_PY_CONTENT = textwrap.dedent("""
    import matplotlib.pyplot as plt

    def plot_rewards(rewards, save_path):
        plt.figure(figsize=(12, 6))
        plt.plot(rewards)
        plt.title('Average Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.grid(True)
        plt.savefig(save_path)
        print(f"Plot successfully saved to {save_path}")
        # plt.show() # Uncomment to display the plot directly
""")

REQUIREMENTS_TXT_CONTENT = textwrap.dedent("""
    torch
    torch_geometric
    networkx
    numpy
    matplotlib
""")

ENVIRONMENT_YML_CONTENT = textwrap.dedent("""
    name: marl-routing
    channels:
      - pytorch
      - pyg
      - conda-forge
      - defaults
    dependencies:
      - python=3.9
      - pytorch
      - torchvision
      - torchaudio
      - pyg
      - networkx
      - numpy
      - matplotlib
      - ipykernel
""")

NSFNET_JSON_CONTENT = textwrap.dedent("""
    {
        "nodes": [
            {"id": 0}, {"id": 1}, {"id": 2}, {"id": 3}, {"id": 4},
            {"id": 5}, {"id": 6}, {"id": 7}, {"id": 8}, {"id": 9},
            {"id": 10}, {"id": 11}, {"id": 12}, {"id": 13}
        ],
        "links": [
            {"source": 0, "target": 1, "capacity": 1000},
            {"source": 0, "target": 2, "capacity": 1000},
            {"source": 0, "target": 3, "capacity": 1000},
            {"source": 1, "target": 2, "capacity": 1000},
            {"source": 1, "target": 7, "capacity": 1000},
            {"source": 2, "target": 5, "capacity": 1000},
            {"source": 3, "target": 4, "capacity": 1000},
            {"source": 3, "target": 6, "capacity": 1000},
            {"source": 4, "target": 5, "capacity": 1000},
            {"source": 4, "target": 9, "capacity": 1000},
            {"source": 5, "target": 7, "capacity": 1000},
            {"source": 6, "target": 8, "capacity": 1000},
            {"source": 6, "target": 11, "capacity": 1000},
            {"source": 7, "target": 10, "capacity": 1000},
            {"source": 8, "target": 9, "capacity": 1000},
            {"source": 8, "target": 12, "capacity": 1000},
            {"source": 9, "target": 10, "capacity": 1000},
            {"source": 10, "target": 13, "capacity": 1000},
            {"source": 11, "target": 12, "capacity": 1000},
            {"source": 12, "target": 13, "capacity": 1000}
        ]
    }
""")

# --- File creation logic ---
files_and_content = {
    'config.py': CONFIG_PY_CONTENT,
    'network_env.py': NETWORK_ENV_PY_CONTENT,
    'models.py': MODELS_PY_CONTENT,
    'agent.py': AGENT_PY_CONTENT,
    'train.py': TRAIN_PY_CONTENT,
    'ospf_baseline.py': OSPF_BASELINE_PY_CONTENT,
    'utils.py': UTILS_PY_CONTENT,
    'requirements.txt': REQUIREMENTS_TXT_CONTENT,
    'environment.yml': ENVIRONMENT_YML_CONTENT,
    os.path.join('topologies', 'nsfnet.json'): NSFNET_JSON_CONTENT,
}

def create_project():
    """Creates the project directory and all necessary files."""
    if os.path.exists(ROOT_DIR):
        print(f"Error: Directory '{ROOT_DIR}' already exists.")
        print("Please remove or rename it before running this script.")
        return

    print(f"Creating project directory: {ROOT_DIR}")
    os.makedirs(ROOT_DIR)
    os.makedirs(os.path.join(ROOT_DIR, 'results'))
    os.makedirs(os.path.join(ROOT_DIR, 'topologies'))

    for file_path, content in files_and_content.items():
        full_path = os.path.join(ROOT_DIR, file_path)
        print(f"  Creating file: {full_path}")
        with open(full_path, 'w') as f:
            f.write(content)

    print("\\n" + "="*50)
    print("PROJECT SUCCESSFULLY CREATED!")
    print("="*50 + "\\n")
    print("NEXT STEPS:\\n")
    print(f"1. Navigate into your new project directory:")
    print(f"   cd {ROOT_DIR}\\n")
    print(f"2. Create the Anaconda environment:")
    print(f"   conda env create -f environment.yml\\n")
    print(f"3. Activate the environment:")
    print(f"   conda activate marl-routing\\n")
    print(f"4. Run the training script to start learning:")
    print(f"   python train.py\\n")
    print(f"5. After training, run the baseline for comparison:")
    print(f"   python ospf_baseline.py\\n")
    print("You can find plots and saved models in the 'results' directory.")
    print("Good luck with your research!")

if __name__ == '__main__':
    create_project()