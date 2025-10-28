import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

class PolicyGNN(nn.Module):
    # --- IMPROVEMENT: This class is now solely the neural network model
    def __init__(self, in_channels, hidden_channels, num_layers, num_agents, agent_partitions):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            
        # --- IMPROVEMENT: Create separate Actor/Critic heads for each agent
        # This allows each agent to specialize its policy and value estimation
        self.actor_heads = nn.ModuleList()
        self.critic_heads = nn.ModuleList()
        
        for i in range(num_agents):
            num_links_for_agent = len(agent_partitions[i])
            if num_links_for_agent == 0:
                self.actor_heads.append(None) # Placeholder if agent has no links
                self.critic_heads.append(None)
                continue

            from config import ACTION_SPACE_SIZE
            # Actor head outputs logits for each link controlled by the agent
            actor_head = nn.Linear(hidden_channels * 2, num_links_for_agent * ACTION_SPACE_SIZE)
            # Critic head outputs a single value for the agent's state
            critic_head = nn.Linear(hidden_channels * 2, 1)

            self.actor_heads.append(actor_head)
            self.critic_heads.append(critic_head)
        
        self.agent_partitions = agent_partitions

    def forward(self, pyg_data):
        x, edge_index = pyg_data.x, pyg_data.edge_index
        
        # 1. Get node embeddings from GNN
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        
        all_agent_logits = []
        all_agent_values = []
        
        # 2. Process through each agent's heads
        for agent_id, actor_head in enumerate(self.actor_heads):
            critic_head = self.critic_heads[agent_id]
            if actor_head is None: continue

            # Create a representation for the agent's subgraph
            agent_nodes = torch.tensor(list(set(n for link in self.agent_partitions[agent_id] for n in link)))
            
            # --- IMPROVEMENT: Better agent state representation (mean and max of node embeddings)
            agent_node_embeddings = x[agent_nodes]
            mean_embedding = agent_node_embeddings.mean(dim=0)
            max_embedding, _ = agent_node_embeddings.max(dim=0)
            agent_embedding = torch.cat([mean_embedding, max_embedding])

            logits = actor_head(agent_embedding)
            value = critic_head(agent_embedding)
            
            all_agent_logits.append(logits)
            all_agent_values.append(value)
            
        return all_agent_logits, all_agent_values

    @staticmethod
    def to_pyg_data(graph, node_features):
        # Convert networkx graph to PyG Data object
        pyg_graph = from_networkx(graph)
        edge_index = pyg_graph.edge_index
        x = torch.tensor(node_features, dtype=torch.float)
        return Data(x=x, edge_index=edge_index)