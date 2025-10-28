import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np
import random

class NetworkRoutingEnv(gym.Env):
    """
    A custom Gymnasium environment for network traffic routing.
    
    The agent's goal is to route packets from their source to their destination
    with minimal latency. Latency is modeled as a combination of transmission
    delay and queuing delay at each node.

    Observation Space:
        - A dictionary containing:
            - 'current_node': The node where the packet is currently located.
            - 'destination_node': The final destination of the packet.
            - 'link_queues': A vector representing the queue size of each link in the network.

    Action Space:
        - A discrete space representing the choice of the next hop (a neighboring node).
          The size of this space is the maximum degree of the network graph.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, graph_file, max_queue_size=50, traffic_rate=0.1):
        super(NetworkRoutingEnv, self).__init__()

        # Load the network topology
        self.graph = nx.read_gml(graph_file, label='id')
        self.nodes = list(self.graph.nodes())
        self.num_nodes = len(self.nodes)
        self.edges = list(self.graph.edges())
        self.num_edges = len(self.edges)
        
        # Environment parameters
        self.max_queue_size = max_queue_size
        self.traffic_rate = traffic_rate # Probability of a new packet spawning at each step

        # Link properties: queue for each directed edge
        self.link_queues = {edge: 0 for edge in self.get_all_directed_edges()}

        # The current packet being routed by the agent
        self.current_packet = None

        # --- Define action and observation spaces ---
        max_degree = max(dict(self.graph.degree()).values())
        self.action_space = spaces.Discrete(max_degree)
        
        self.observation_space = spaces.Dict({
            'current_node': spaces.Discrete(self.num_nodes),
            'destination_node': spaces.Discrete(self.num_nodes),
            'link_queues': spaces.Box(low=0, high=self.max_queue_size, 
                                      shape=(self.num_edges * 2,), dtype=np.float32)
        })

    def get_all_directed_edges(self):
        """Returns a list of all directed edges (u,v) and (v,u)."""
        directed_edges = []
        for u, v in self.edges:
            directed_edges.append((u, v))
            directed_edges.append((v, u))
        return directed_edges

    def _get_obs(self):
        """Constructs the observation dictionary from the current state."""
        queue_vector = np.array(list(self.link_queues.values()), dtype=np.float32)
        
        return {
            'current_node': self.current_packet['current'],
            'destination_node': self.current_packet['destination'],
            'link_queues': queue_vector
        }

    def _generate_packet(self):
        """Generates a new packet with a random source and destination."""
        source, destination = random.sample(self.nodes, 2)
        self.current_packet = {
            'source': source,
            'destination': destination,
            'current': source,
            'path': [source],
            'latency': 0
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Clear all queues
        for edge in self.link_queues:
            self.link_queues[edge] = 0
            
        # Create the first packet to be routed
        self._generate_packet()
        
        observation = self._get_obs()
        info = {} # No extra info on reset
        
        return observation, info

    def step(self, action):
        """
        Executes one time step within the environment.
        The action corresponds to choosing a neighbor to forward the current packet.
        """
        current_node = self.current_packet['current']
        neighbors = list(self.graph.neighbors(current_node))
        
        # --- 1. Determine the action and its validity ---
        if action >= len(neighbors):
            # Invalid action: agent chose a non-existent neighbor.
            # This is a strong penalty.
            return self._get_obs(), -100, True, False, {}

        next_hop = neighbors[action]
        
        # --- 2. Update network state ---
        # Update packet's state
        self.current_packet['path'].append(next_hop)
        self.current_packet['current'] = next_hop
        
        # Calculate latency for this hop
        # Latency = Queuing Delay + Transmission Delay (simplified to 1)
        link = (current_node, next_hop)
        queuing_delay = self.link_queues[link]
        transmission_delay = 1 
        latency_this_hop = queuing_delay + transmission_delay
        self.current_packet['latency'] += latency_this_hop

        # Update the queue of the chosen link
        if self.link_queues[link] < self.max_queue_size:
            self.link_queues[link] += 1
        
        # Simulate other traffic and queue reduction
        self._simulate_background_traffic()

        # --- 3. Determine reward, done, and truncated ---
        terminated = False
        reward = -latency_this_hop # Negative reward to encourage minimizing latency

        if next_hop == self.current_packet['destination']:
            # Packet reached destination
            reward += 50  # Bonus for successful delivery
            terminated = True
        
        if len(self.current_packet['path']) > self.num_nodes * 2:
            # Packet is likely in a loop, terminate
            reward -= 50 # Penalty for looping
            terminated = True
        
        # After an episode ends (terminated), we generate a new packet for the next episode
        if terminated:
            self._generate_packet()

        observation = self._get_obs()
        info = {'path': self.current_packet['path'], 'total_latency': self.current_packet['latency']}
        truncated = False # We don't use truncation in this simple version

        return observation, reward, terminated, truncated, info

    def _simulate_background_traffic(self):
        """
        Simulates the reduction of queues over time and adds new background traffic.
        """
        # Decrease all queues by a fixed service rate
        for link in self.link_queues:
            self.link_queues[link] = max(0, self.link_queues[link] - 1) # Service rate of 1 packet/step

        # Occasionally add a new packet to a random queue
        if random.random() < self.traffic_rate:
            source, dest = random.sample(self.nodes, 2)
            if self.graph.has_edge(source, dest):
                link = (source, dest)
                if self.link_queues[link] < self.max_queue_size:
                    self.link_queues[link] += 1

if __name__ == '__main__':
    # --- Test Script ---
    print("Testing NetworkRoutingEnv...")
    
    # Use the relative path to the topology file
    # This assumes you are running the script from the root of the project folder
    # e.g., python synapse/network_env.py
    try:
        env = NetworkRoutingEnv(graph_file='data/topologies/nsfnet.gml')
    except FileNotFoundError:
        print("\nERROR: nsfnet.gml not found.")
        print("Please ensure you are running this script from the project's root directory.")
        print("Example: `python synapse/network_env.py`\n")
        exit()

    # Test reset
    obs, info = env.reset()
    print("Environment created and reset successfully.")
    print(f"Initial Observation Keys: {obs.keys()}")
    print(f"Current Packet Source: {env.current_packet['source']}")
    print(f"Current Packet Destination: {env.current_packet['destination']}")

    # Test a few random steps
    print("\nSimulating 5 random steps...")
    for i in range(5):
        action = env.action_space.sample() # Take a random action
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {i+1}:")
        print(f"  - Action taken: {action}")
        print(f"  - Packet at node: {obs['current_node']}")
        print(f"  - Reward received: {reward:.2f}")
        print(f"  - Terminated: {terminated}")
        print(f"  - Path so far: {info.get('path', [])}")

        if terminated:
            print("  - Episode finished. A new packet was generated.")
            print(f"  - New Packet Source: {env.current_packet['source']}")
            print(f"  - New Packet Destination: {env.current_packet['destination']}")

    env.close()
    print("\nEnvironment test complete.")