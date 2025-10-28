import torch
import numpy as np
from network_env import NetworkEnv
from agent import AgentManager
from models import PolicyGNN
import config
# --- IMPROVEMENT: Import the learning rate scheduler ---
from torch.optim.lr_scheduler import StepLR

def main():
    print("--- Initializing Environment and Agent Manager ---")
    env = NetworkEnv(config.TOPOLOGY_FILE, config.NUM_AGENTS)
    manager = AgentManager(env)
    
    # --- IMPROVEMENT: Create the learning rate scheduler ---
    # This will automatically adjust the learning rate during training
    # based on the parameters set in config.py.
    scheduler = StepLR(manager.optimizer, 
                       step_size=config.LR_SCHEDULER_STEP_SIZE, 
                       gamma=config.LR_SCHEDULER_GAMMA)

    episode_rewards = []
    print(f"--- Starting Training for {config.NUM_EPISODES} Episodes ---")

    for i_episode in range(config.NUM_EPISODES):
        state = env.reset()
        episode_reward_sum = 0
        
        # Inner loop for the steps within an episode
        for t in range(config.MAX_STEPS_PER_EPISODE):
            pyg_data = PolicyGNN.to_pyg_data(env.graph, state)
            
            # The select_actions method computes and stores values, log_probs, etc.
            actions = manager.select_actions(pyg_data)

            next_state, reward, done, _ = env.step(actions)
            
            # Store the global reward for this step
            manager.rewards.append(reward)
            
            state = next_state
            episode_reward_sum += reward
            
            # Perform a policy update every N_STEPS
            if (t + 1) % config.N_STEPS == 0:
                next_pyg_data = PolicyGNN.to_pyg_data(env.graph, next_state)
                manager.update(next_pyg_data)
                # The manager's update function now clears its own buffers

        # --- IMPROVEMENT: Step the scheduler at the end of each episode ---
        # This will decay the learning rate at the specified intervals.
        scheduler.step()

        # Log the average reward for the completed episode
        episode_rewards.append(episode_reward_sum / config.MAX_STEPS_PER_EPISODE)

        # Print logs at the specified interval
        if (i_episode + 1) % config.LOG_INTERVAL == 0:
            # Get the current learning rate to display in the logs
            current_lr = scheduler.get_last_lr()[0]
            avg_reward = np.mean(episode_rewards[-config.LOG_INTERVAL:])
            
            # --- IMPROVEMENT: Added Learning Rate to the log output ---
            print(f'Episode {i_episode+1}/{config.NUM_EPISODES}\tAvg reward: {avg_reward:.3f}\tLR: {current_lr:.1e}')

    print("--- Training Complete ---")
    
    # Use distinct names for the final model and plot to avoid overwriting
    final_model_path = config.MODEL_SAVE_PATH.replace('.pth', '_final.pth')
    final_plot_path = config.PLOT_SAVE_PATH.replace('.png', '_final.png')

    torch.save(manager.policy.state_dict(), final_model_path)
    print(f"Model saved to {final_model_path}")

    # Plot and save the training curve
    from utils import plot_rewards
    plot_rewards(episode_rewards, save_path=final_plot_path)

if __name__ == '__main__':
    main()