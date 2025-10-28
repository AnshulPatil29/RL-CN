import os
import time
from synapse.network_env import NetworkRoutingEnv
from synapse.agent import CustomNetworkFeatureExtractor
from stable_baselines3 import PPO

class MARLSystem:
    """
    Manages the training of the shared PPO agent for network routing.
    
    This system implements a "parameter sharing" multi-agent approach.
    Conceptually, there is an agent at each node, but they all share the same
    underlying policy network and contribute to its training.
    """
    def __init__(self, env_config, agent_config, log_dir="./logs/"):
        """
        Initializes the MARL system.
        
        Args:
            env_config (dict): Configuration for the network environment.
            agent_config (dict): Configuration for the PPO agent.
            log_dir (str): Directory to save logs and models.
        """
        self.env = NetworkRoutingEnv(**env_config)
        self.agent_config = agent_config
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        policy_kwargs = {
            'features_extractor_class': CustomNetworkFeatureExtractor,
            'features_extractor_kwargs': {'features_dim': self.agent_config.get('features_dim', 128)},
        }
        
        self.model = PPO(
            "MultiInputPolicy",
            self.env,
            policy_kwargs=policy_kwargs,
            n_steps=self.agent_config.get('n_steps', 2048),
            batch_size=self.agent_config.get('batch_size', 64),
            gamma=self.agent_config.get('gamma', 0.99),
            learning_rate=self.agent_config.get('learning_rate', 0.0003),
            verbose=1,
            tensorboard_log=os.path.join(self.log_dir, "tensorboard")
        )

    def train(self, total_timesteps):
        """
        Trains the shared agent for a given number of timesteps.
        """
        print("--- Starting Training ---")
        start_time = time.time()
        
        self.model.learn(
            total_timesteps=total_timesteps,
            progress_bar=True
        )
        
        end_time = time.time()
        print(f"--- Training Finished ---")
        print(f"Total training time: {(end_time - start_time) / 60:.2f} minutes")

    def save_model(self, model_name="synapse_marl_model.zip"):
        """Saves the trained model to a file."""
        model_path = os.path.join(self.log_dir, model_name)
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path):
        """Loads a pre-trained model."""
        self.model = PPO.load(model_path, env=self.env)
        print(f"Model loaded from {model_path}")

if __name__ == '__main__':
    # --- Test Script ---
    print("Testing the MARLSystem training pipeline...")

    # Configuration for the environment and agent
    env_params = {
        'graph_file': 'data/topologies/nsfnet.gml',
        'traffic_rate': 0.2
    }
    
    agent_params = {
        'features_dim': 128,
        'n_steps': 256, # Use a small value for a quick test
        'batch_size': 64,
        'learning_rate': 0.0005
    }
    
    # Create the MARL system
    marl_system = MARLSystem(
        env_config=env_params,
        agent_config=agent_params,
        log_dir="./logs/test_run/"
    )
    print("MARLSystem initialized successfully.")

    # Run a very short training loop to ensure everything works
    # A real training run would be millions of timesteps.
    test_timesteps = 1000
    print(f"\nStarting a short training run for {test_timesteps} timesteps...")
    
    try:
        marl_system.train(total_timesteps=test_timesteps)
    except Exception as e:
        print(f"An error occurred during the test training run: {e}")
        print("Test FAILED.")
        # Add a specific hint for file not found
        if isinstance(e, FileNotFoundError):
             print("\nHint: Ensure you are running this script from the project's root directory.")
             print("Example: `python synapse/marl_system.py`\n")
        exit()

    print("\nShort training run completed without errors.")

    # Test saving the model
    marl_system.save_model("test_model.zip")
    
    # Test loading the model
    try:
        marl_system.load_model("./logs/test_run/test_model.zip")
        print("Model loading test PASSED.")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        print("Model loading test FAILED.")
        exit()
        
    print("\nMARLSystem test PASSED.")