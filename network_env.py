import networkx as nx
import numpy as np
import json
from networkx.readwrite import json_graph

class NetworkEnv:
    def __init__(self, topology_file, num_agents):
        self.topology_file = topology_file
        self.num_agents = num_agents
        self.graph = self._load_topology()
        self.nodes = sorted(list(self.graph.nodes())) # --- IMPROVEMENT: Sort nodes for consistency
        self.num_nodes = len(self.nodes)
        self.num_links = self.graph.number_of_edges()
        self.agent_link_partition = self._partition_links()
        self.traffic_matrix = None

    def _load_topology(self):
        with open(self.topology_file) as f:
            data = json.load(f)
        
        # --- IMPROVEMENT: Let networkx handle node types properly
        G = json_graph.node_link_graph(data)
        
        # Ensure graph is treated as undirected
        G_undirected = nx.Graph(G)
        
        nx.set_edge_attributes(G_undirected, 1.0, 'weight')
        nx.set_edge_attributes(G_undirected, 0.0, 'utilization')
        return G_undirected
    
    def _partition_links(self):
        partition = {i: [] for i in range(self.num_agents)}
        # --- IMPROVEMENT: Sort edges for deterministic partitioning across runs
        links = sorted([tuple(sorted(e)) for e in self.graph.edges()])

        for idx, link in enumerate(links):
            agent_id = idx % self.num_agents
            partition[agent_id].append(link)
        return partition

    # In network_env.py, replace only the _generate_traffic_matrix method.

    def _generate_traffic_matrix(self, model='gravity', demand_multiplier=1.0, hotspot=None):
        """
        Generates a traffic matrix with more control for scenarios.
        
        Args:
            model (str): The base traffic model ('gravity' or 'random').
            demand_multiplier (float): A factor to scale all traffic demands.
            hotspot (tuple): A tuple of (src, dst, demand) to add a specific high-demand flow.
        """
        self.traffic_matrix = {}
        for src in self.nodes:
            for dst in self.nodes:
                if src != dst:
                    if model == 'gravity':
                        demand = np.random.uniform(5, 20) * np.random.uniform(0.5, 1.5)
                    else:
                        demand = np.random.uniform(1, 50) if np.random.rand() > 0.5 else 0
                    
                    if demand > 0:
                        self.traffic_matrix[(src, dst)] = demand * demand_multiplier
        
        if hotspot:
            src, dst, demand = hotspot
            # Add or overwrite the hotspot demand
            self.traffic_matrix[(src, dst)] = demand * demand_multiplier
    
    def reset(self):
        nx.set_edge_attributes(self.graph, 0.0, 'utilization')
        self._generate_traffic_matrix()
        return self._get_state()

    # In network_env.py, replace only the step method with this one.

    def step(self, actions):
        # 1. Apply actions (update link weights)
        for agent_id, agent_actions in actions.items():
            for i, (u, v) in enumerate(self.agent_link_partition[agent_id]):
                
                # --- THIS IS THE FIX ---
                # Check if the link still exists in the graph before trying to modify it.
                # This makes the environment robust to link failures.
                if self.graph.has_edge(u, v):
                    # Action is an index from 0 to N-1, weight is action + 1
                    weight = agent_actions[i] + 1
                    self.graph[u][v]['weight'] = weight
                # --- END OF FIX ---

        # 2. Route traffic and update utilization
        nx.set_edge_attributes(self.graph, 0.0, 'utilization')
        total_latency, total_traffic = 0, 0

        for (src, dst), demand in self.traffic_matrix.items():
            try:
                path = nx.shortest_path(self.graph, source=src, target=dst, weight='weight')
                path_latency = len(path) - 1
                total_latency += path_latency * demand
                total_traffic += demand
                
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    self.graph[u][v]['utilization'] += demand
            except (nx.NetworkXNoPath, KeyError):
                continue

        # 3. Calculate reward and next state
        reward = self._get_reward(total_latency, total_traffic)
        self._generate_traffic_matrix() # New traffic for the next state
        next_state = self._get_state()
        
        return next_state, reward, False, {}
    
    def _get_state(self):
        node_features = np.zeros((self.num_nodes, 2))
        util_sum = {node: 0.0 for node in self.nodes}

        for u, v, attr in self.graph.edges(data=True):
            norm_util = min(attr.get('utilization', 0.0) / attr.get('capacity', 1.0), 1.0)
            util_sum[u] += norm_util
            util_sum[v] += norm_util
        
        for i, node in enumerate(self.nodes):
            degree = self.graph.degree[node]
            if degree > 0:
                node_features[i, 0] = util_sum[node] / degree
            node_features[i, 1] = degree / self.num_nodes # Normalize degree
            
        return node_features

    def _get_reward(self, total_latency, total_traffic):
        from config import REWARD_WEIGHTS
        utilizations = [attr['utilization'] / attr['capacity'] for _, _, attr in self.graph.edges(data=True)]
        max_utilization = max(utilizations) if utilizations else 0
        avg_latency = total_latency / total_traffic if total_traffic > 0 else 0
        
        reward_util = REWARD_WEIGHTS['utilization'] * max_utilization**2 # Penalize high util more
        reward_latency = REWARD_WEIGHTS['latency'] * avg_latency
        return reward_util + reward_latency