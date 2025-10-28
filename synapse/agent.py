# FULL REPLACEMENT FOR synapse/agent.py (BATCH MISMATCH FIX)

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_checker import check_env

# Assuming network_env.py is in the same directory (synapse)
from .network_env import NetworkRoutingEnv


class CustomFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for the network routing environment.
    This version handles inconsistencies in batch sizes between different observation keys.
    """
    def __init__(self, observation_space: gym.spaces.Dict, embedding_dim: int = 32):
        super().__init__(observation_space, features_dim=1) 
        self.num_nodes = observation_space['current_node'].n
        
        self.node_embedding = nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=embedding_dim)

        link_queues_shape = observation_space['link_queues'].shape[0]
        self.queue_processor = nn.Sequential(
            nn.Linear(link_queues_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        self._features_dim = (embedding_dim * 2) + 32
        
    def forward(self, observations: dict) -> torch.Tensor:
        """
        Forward pass of the feature extractor.
        """
        # 1. Process node observations.
        current_node_obs = observations['current_node'].long().flatten()
        destination_node_obs = observations['destination_node'].long().flatten()
        
        current_node_embedding = self.node_embedding(current_node_obs)
        destination_node_embedding = self.node_embedding(destination_node_obs)
        
        # 2. Process queue observations.
        processed_queues = self.queue_processor(observations['link_queues'])
        
        # --- ROBUSTNESS FIX FOR BATCH SIZE MISMATCH ---
        # Get the batch size from the node embeddings (which SB3 might have expanded)
        node_batch_size = current_node_embedding.shape[0]
        queue_batch_size = processed_queues.shape[0]

        # If queue tensor's batch size is 1 while node's is > 1, expand it.
        # This treats the queue state as a "global context" for all nodes.
        if node_batch_size > 1 and queue_batch_size == 1:
            processed_queues = processed_queues.expand(node_batch_size, -1)
        # --- END OF FIX ---

        # 3. Concatenate all feature vectors.
        final_features = torch.cat([current_node_embedding, destination_node_embedding, processed_queues], dim=1)
        
        return final_features

def create_agent(env):
    """
    Creates a PPO agent with the custom feature extractor.
    """
    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor,
        features_extractor_kwargs=dict(embedding_dim=32),
        net_arch=dict(pi=[64, 64], vf=[64, 64])
    )

    agent = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./tensorboard_logs/synapse_ppo/"
    )
    return agent

if __name__ == '__main__':
    print("This file contains the agent definition. To train, run the notebooks.")
    pass