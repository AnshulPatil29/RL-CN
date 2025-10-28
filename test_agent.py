import os
import gymnasium as gym
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomNetworkFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for our dictionary observation space.
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 64):
        super().__init__(observation_space, features_dim)

        extractors = {}
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            if key in ["current_node", "destination_node"]:
                embedding_dim = 16
                extractor = nn.Embedding(subspace.n, embedding_dim)
                total_concat_size += embedding_dim
                extractors[key] = extractor
            elif key == "link_queues":
                extractor = nn.Sequential(
                    nn.Linear(subspace.shape[0], 64), nn.ReLU(),
                    nn.Linear(64, 32), nn.ReLU(),
                )
                total_concat_size += 32
                extractors[key] = extractor
        
        self.extractors = nn.ModuleDict(extractors)
        self.fc = nn.Sequential(nn.Linear(total_concat_size, features_dim), nn.ReLU())

    def forward(self, observations: dict) -> torch.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            tensor_input = observations[key]
            
            if key in ["current_node", "destination_node"]:
                # The input is (batch_size, 1), cast to long
                tensor_input = tensor_input.long()
                embedded = extractor(tensor_input)
                # The output is (batch_size, 1, embedding_dim), squeeze to (batch_size, embedding_dim)
                encoded_tensor_list.append(embedded.squeeze(1))
            else: # for 'link_queues'
                # The input is already (batch_size, num_queues)
                encoded_tensor_list.append(extractor(tensor_input))
        
        return self.fc(torch.cat(encoded_tensor_list, dim=1))

# Keep the __main__ script for basic, independent testing of this file.
if __name__ == '__main__':
    print("Testing the CustomNetworkFeatureExtractor and Agent Policy...")
    from synapse.network_env import NetworkRoutingEnv

    topology_file = 'data/topologies/nsfnet.gml'
    if not os.path.exists(topology_file):
        print(f"\nERROR: Please run from the project root. CWD: {os.getcwd()}")
        exit()
        
    env = NetworkRoutingEnv(graph_file=topology_file)
    policy_kwargs = {'features_extractor_class': CustomNetworkFeatureExtractor}

    try:
        agent = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, n_steps=128, verbose=0)
        print("\n✅ Successfully created a PPO agent with the custom policy.")
    except Exception as e:
        print(f"\n❌ An error occurred while creating the agent: {e}")
        import traceback; traceback.print_exc()
        print("Test FAILED.")
        exit()

    print("\nSimulating a single prediction step...")
    obs, _ = env.reset()
    action, _states = agent.predict(obs, deterministic=True)
    print(f"✅ Successfully predicted an action: {action}")
    print("\n✅ Custom agent architecture test PASSED.")