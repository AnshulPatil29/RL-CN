import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

import config
from network_env import NetworkEnv
from agent import AgentManager
from models import PolicyGNN

# --- Evaluation Configuration ---
# Use the final model saved by the scheduled training script
MODEL_TO_EVALUATE = config.MODEL_SAVE_PATH.replace('.pth', '_final.pth')
EVALUATION_STEPS = 500  # Number of different traffic matrices to test on
PLOT_SAVE_PATH = 'results/evaluation_comparison.png'


def calculate_step_metrics(env):
    """
    Calculates the max link utilization and average path latency for the current
    state of the environment's graph. This is a fair, unified way to measure
    performance for any agent.
    """
    # 1. Calculate Max Link Utilization
    utilizations = [
        attr['utilization'] / attr['capacity'] 
        for _, _, attr in env.graph.edges(data=True) if attr['capacity'] > 0
    ]
    max_utilization = max(utilizations) if utilizations else 0

    # 2. Calculate Average Path Latency
    total_latency = 0
    total_traffic = 0
    for (src, dst), demand in env.traffic_matrix.items():
        try:
            # Re-calculate the path taken to be sure, using the current weights
            path = nx.shortest_path(env.graph, source=src, target=dst, weight='weight')
            path_latency = len(path) - 1
            total_latency += path_latency * demand
            total_traffic += demand
        except (nx.NetworkXNoPath, KeyError):
            continue
    
    avg_latency = total_latency / total_traffic if total_traffic > 0 else 0
    
    return max_utilization, avg_latency


def evaluate_rl_agent(env, manager):
    """
    Evaluates the trained MARL agent.
    """
    print(f"--- Evaluating Trained RL Agent from {MODEL_TO_EVALUATE} ---")
    
    # Load the trained policy weights
    try:
        manager.policy.load_state_dict(torch.load(MODEL_TO_EVALUATE))
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {MODEL_TO_EVALUATE}")
        print("Please run train.py with the LR scheduler to generate the final model.")
        return None
        
    # Set the model to evaluation mode (important for some layers like dropout)
    manager.policy.eval()

    metrics = {'max_utilization': [], 'avg_latency': []}
    
    state = env.reset()
    
    # Use torch.no_grad() for inference as we are not training
    with torch.no_grad():
        for _ in tqdm(range(EVALUATION_STEPS), desc="RL Agent Evaluation"):
            pyg_data = PolicyGNN.to_pyg_data(env.graph, state)
            actions = manager.select_actions(pyg_data)
            
            # The env.step function applies the actions and routes the traffic
            # We don't need the reward, just the resulting graph state
            next_state, _, _, _ = env.step(actions)
            
            # Calculate metrics based on the outcome of the agent's actions
            max_util, avg_lat = calculate_step_metrics(env)
            metrics['max_utilization'].append(max_util)
            metrics['avg_latency'].append(avg_lat)
            
            state = next_state

    # Return the average of the collected metrics
    return {
        "Agent": "MARL (Synapse)",
        "Avg Max Utilization": np.mean(metrics['max_utilization']),
        "Avg Path Latency": np.mean(metrics['avg_latency'])
    }


def evaluate_ospf_baseline(env):
    """
    Evaluates the OSPF (shortest-path) baseline.
    """
    print("\n--- Evaluating OSPF Baseline ---")

    metrics = {'max_utilization': [], 'avg_latency': []}
    
    # For OSPF, weights are static (typically all 1s for shortest hop count)
    nx.set_edge_attributes(env.graph, 1, 'weight')
    
    env.reset()

    for _ in tqdm(range(EVALUATION_STEPS), desc="OSPF Baseline Evaluation"):
        # We don't call env.step() because actions are fixed.
        # We manually simulate the OSPF process for each traffic matrix.

        # 1. Reset utilization on the graph
        nx.set_edge_attributes(env.graph, 0.0, 'utilization')
        
        # 2. Route all traffic demands using the static weights
        for (src, dst), demand in env.traffic_matrix.items():
            try:
                path = nx.shortest_path(env.graph, source=src, target=dst, weight='weight')
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    env.graph[u][v]['utilization'] += demand
            except (nx.NetworkXNoPath, KeyError):
                continue
        
        # 3. Calculate metrics for this step
        max_util, avg_lat = calculate_step_metrics(env)
        metrics['max_utilization'].append(max_util)
        metrics['avg_latency'].append(avg_lat)

        # 4. Generate a new traffic matrix for the next evaluation step
        env._generate_traffic_matrix()

    return {
        "Agent": "OSPF (Shortest Path)",
        "Avg Max Utilization": np.mean(metrics['max_utilization']),
        "Avg Path Latency": np.mean(metrics['avg_latency'])
    }


def plot_results(results_df):
    """
    Generates and saves a bar chart comparing the performance of the agents.
    """
    print(f"\n--- Generating Comparison Plot ---")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Average Max Link Utilization
    sns.barplot(x="Agent", y="Avg Max Utilization", data=results_df, ax=axes[0], palette="viridis")
    axes[0].set_title("Congestion Performance (Lower is Better)", fontsize=14)
    axes[0].set_ylabel("Average Maximum Link Utilization", fontsize=12)
    axes[0].set_xlabel("")

    # Plot 2: Average Path Latency
    sns.barplot(x="Agent", y="Avg Path Latency", data=results_df, ax=axes[1], palette="plasma")
    axes[1].set_title("Latency Performance (Lower is Better)", fontsize=14)
    axes[1].set_ylabel("Average Path Latency", fontsize=12)
    axes[1].set_xlabel("")
    
    plt.suptitle("MARL Agent vs. OSPF Baseline Performance", fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure
    plt.savefig(PLOT_SAVE_PATH)
    print(f"Comparison plot saved to: {PLOT_SAVE_PATH}")
    # plt.show() # Uncomment to display the plot directly


def main():
    # Initialize the environment and the agent manager
    env = NetworkEnv(config.TOPOLOGY_FILE, config.NUM_AGENTS)
    manager = AgentManager(env)
    
    # Run evaluations
    rl_results = evaluate_rl_agent(env, manager)
    if rl_results is None:
        return # Stop if the model file wasn't found
        
    ospf_results = evaluate_ospf_baseline(env)
    
    # Combine results into a pandas DataFrame for easy plotting and display
    results_df = pd.DataFrame([rl_results, ospf_results])
    
    print("\n" + "="*50)
    print("           PERFORMANCE COMPARISON RESULTS")
    print("="*50)
    print(results_df.to_string(index=False))
    print("="*50)
    
    # Generate and save the final comparison plot
    plot_results(results_df)

if __name__ == '__main__':
    main()