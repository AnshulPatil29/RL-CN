import torch
import numpy as np
from network_env import NetworkEnv
from agent import AgentManager
from models import PolicyGNN
import config

def main():
    print("--- Initializing Environment and Agent Manager ---")
    env = NetworkEnv(config.TOPOLOGY_FILE, config.NUM_AGENTS)
    manager = AgentManager(env)
    
    episode_rewards = []
    print(f"--- Starting Training for {config.NUM_EPISODES} Episodes ---")

    for i_episode in range(config.NUM_EPISODES):
        state = env.reset()
        episode_reward_sum = 0
        
        ## --- IMPROVEMENT: A2C-style training loop ---
        # In train.py, replace the main loop inside the main() function

        ## --- IMPROVEMENT: A2C-style training loop ---
        for t in range(config.MAX_STEPS_PER_EPISODE):
            pyg_data = PolicyGNN.to_pyg_data(env.graph, state)
            
            # The select_actions method now also computes and stores the value for the current state
            actions = manager.select_actions(pyg_data)

            next_state, reward, done, _ = env.step(actions)
            
            # Store experience for the A2C update
            manager.rewards.append(reward)
            
            state = next_state
            episode_reward_sum += reward
            
            # Perform an update every N_STEPS
            if (t + 1) % config.N_STEPS == 0:
                next_pyg_data = PolicyGNN.to_pyg_data(env.graph, next_state)
                manager.update(next_pyg_data)
                # After updating, the manager clears its own buffers

        episode_rewards.append(episode_reward_sum / config.MAX_STEPS_PER_EPISODE)

        if (i_episode + 1) % config.LOG_INTERVAL == 0:
            avg_reward = np.mean(episode_rewards[-config.LOG_INTERVAL:])
            print(f'Episode {i_episode+1}/{config.NUM_EPISODES}\tAverage reward: {avg_reward:.3f}')

    print("--- Training Complete ---")
    
    torch.save(manager.policy.state_dict(), config.MODEL_SAVE_PATH)
    print(f"Model saved to {config.MODEL_SAVE_PATH}")

    from utils import plot_rewards
    plot_rewards(episode_rewards, save_path=config.PLOT_SAVE_PATH)

if __name__ == '__main__':
    main()