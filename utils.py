
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
