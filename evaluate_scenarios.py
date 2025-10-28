import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import random

import config
from network_env import NetworkEnv
from agent import AgentManager
from models import PolicyGNN

# --- Evaluation Configuration ---
MODEL_TO_EVALUATE = config.MODEL_SAVE_PATH.replace('.pth', '_final.pth')
EVALUATION_STEPS = 200  # Number of steps per scenario
PLOT_SAVE_PATH = 'results/scenario_evaluation_comparison.png'


def calculate_step_metrics(env):
    """Calculates max link utilization and average path latency for the current graph state."""
    utilizations = [
        attr['utilization'] / attr['capacity'] 
        for _, _, attr in env.graph.edges(data=True) if attr.get('capacity', 0) > 0
    ]
    max_utilization = max(utilizations) if utilizations else 0

    total_latency, total_traffic = 0, 0
    for (src, dst), demand in env.traffic_matrix.items():
        try:
            path = nx.shortest_path(env.graph, source=src, target=dst, weight='weight')
            total_latency += (len(path) - 1) * demand
            total_traffic += demand
        except (nx.NetworkXNoPath, KeyError):
            continue
    
    avg_latency = total_latency / total_traffic if total_traffic > 0 else 0
    return max_utilization, avg_latency


def run_evaluation_for_agent(agent_name, env, manager=None):
    """A unified evaluation loop for any agent (RL or OSPF)."""
    metrics = {'max_utilization': [], 'avg_latency': []}
    
    # Set the policy for the agent being evaluated
    if agent_name == 'MARL (Synapse)':
        manager.policy.eval()
    else: # OSPF
        nx.set_edge_attributes(env.graph, 1, 'weight')

    # The main evaluation loop
    for _ in tqdm(range(EVALUATION_STEPS), desc=f"Evaluating {agent_name}"):
        
        # --- AGENT-SPECIFIC LOGIC TO UPDATE LINK UTILIZATION ---
        if agent_name == 'MARL (Synapse)':
            with torch.no_grad():
                pyg_data = PolicyGNN.to_pyg_data(env.graph, env._get_state())
                actions = manager.select_actions(pyg_data)
                # The env.step function applies actions and routes all traffic internally
                env.step(actions)
        else: # OSPF
            nx.set_edge_attributes(env.graph, 0.0, 'utilization')
            for (src, dst), demand in env.traffic_matrix.items():
                try:
                    path = nx.shortest_path(env.graph, source=src, target=dst, weight='weight')
                    for i in range(len(path) - 1):
                        # --- THIS IS THE FIX ---
                        # Define u and v from the current path segment before using them
                        u, v = path[i], path[i+1]
                        env.graph[u][v]['utilization'] += demand
                        # --- END OF FIX ---
                except (nx.NetworkXNoPath, KeyError):
                    continue
        
        # --- METRICS CALCULATION (SAME FOR BOTH AGENTS) ---
        max_util, avg_lat = calculate_step_metrics(env)
        metrics['max_utilization'].append(max_util)
        metrics['avg_latency'].append(avg_lat)
        
        # Manually generate the next traffic matrix for the next step of the evaluation
        env._generate_traffic_matrix(**getattr(env, 'scenario_params', {}))

    return {
        "Avg Max Utilization": np.mean(metrics['max_utilization']),
        "Avg Path Latency": np.mean(metrics['avg_latency'])
    }


def plot_scenario_results(results_df):
    """Generates and saves a grouped bar chart comparing performance across scenarios."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    sns.set_theme(style="whitegrid")

    df_util = results_df.melt(id_vars=['Scenario', 'Agent'], value_vars=['Avg Max Utilization'], var_name='Metric', value_name='Value')
    df_lat = results_df.melt(id_vars=['Scenario', 'Agent'], value_vars=['Avg Path Latency'], var_name='Metric', value_name='Value')

    sns.barplot(data=df_util, x='Scenario', y='Value', hue='Agent', ax=axes[0], palette='viridis')
    axes[0].set_title('Congestion Performance (Lower is Better)', fontsize=14)
    axes[0].set_ylabel('Avg. Max Link Utilization', fontsize=12)
    axes[0].set_xlabel('Scenario', fontsize=12)
    axes[0].legend(title='Agent')

    sns.barplot(data=df_lat, x='Scenario', y='Value', hue='Agent', ax=axes[1], palette='plasma')
    axes[1].set_title('Latency Performance (Lower is Better)', fontsize=14)
    axes[1].set_ylabel('Avg. Path Latency (Hops)', fontsize=12)
    axes[1].set_xlabel('Scenario', fontsize=12)
    axes[1].legend(title='Agent')

    plt.suptitle("Performance Comparison Across Network Scenarios", fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(PLOT_SAVE_PATH)
    print(f"\nScenario comparison plot saved to: {PLOT_SAVE_PATH}")


def main():
    # --- Setup ---
    env = NetworkEnv(config.TOPOLOGY_FILE, config.NUM_AGENTS)
    manager = AgentManager(env)
    try:
        manager.policy.load_state_dict(torch.load(MODEL_TO_EVALUATE))
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {MODEL_TO_EVALUATE}. Please run train.py first.")
        return

    all_results = []
    
    # --- Scenario Definitions ---
    scenarios = {
        "Normal Traffic": {},
        "High Congestion": {"demand_multiplier": 2.0},
        "Sudden Hotspot": {"hotspot": (0, 13, 400)},
    }
    
    # --- Run Scenario Evaluations ---
    for scenario_name, scenario_params in scenarios.items():
        print(f"\n{'='*20} SCENARIO: {scenario_name.upper()} {'='*20}")
        
        env.scenario_params = scenario_params
        
        random.seed(42)
        np.random.seed(42)
        env.reset()
        
        rl_metrics = run_evaluation_for_agent('MARL (Synapse)', env, manager)
        rl_metrics.update({"Agent": "MARL (Synapse)", "Scenario": scenario_name})
        all_results.append(rl_metrics)
        
        random.seed(42)
        np.random.seed(42)
        env.reset()

        ospf_metrics = run_evaluation_for_agent('OSPF (Shortest Path)', env)
        ospf_metrics.update({"Agent": "OSPF (Shortest Path)", "Scenario": scenario_name})
        all_results.append(ospf_metrics)

    # --- Link Failure Scenario (Special Case) ---
    print(f"\n{'='*20} SCENARIO: LINK FAILURE {'='*20}")
    
    original_graph = env.graph.copy()
    
    centrality = nx.edge_betweenness_centrality(env.graph)
    critical_link = max(centrality, key=centrality.get)
    u_crit, v_crit = critical_link
    print(f"Simulating failure of critical link: {(u_crit, v_crit)}")
    
    env.graph.remove_edge(u_crit, v_crit)
    env.scenario_params = {}

    random.seed(42)
    np.random.seed(42)
    env.reset()
    
    rl_metrics = run_evaluation_for_agent('MARL (Synapse)', env, manager)
    rl_metrics.update({"Agent": "MARL (Synapse)", "Scenario": "Link Failure"})
    all_results.append(rl_metrics)
    
    random.seed(42)
    np.random.seed(42)
    env.reset()
    
    ospf_metrics = run_evaluation_for_agent('OSPF (Shortest Path)', env)
    ospf_metrics.update({"Agent": "OSPF (Shortest Path)", "Scenario": "Link Failure"})
    all_results.append(ospf_metrics)
    
    env.graph = original_graph

    # --- Display Results ---
    results_df = pd.DataFrame(all_results)
    print("\n" + "="*60)
    print("           SCENARIO PERFORMANCE COMPARISON RESULTS")
    print("="*60)
    print(results_df.to_string())
    print("="*60)

    plot_scenario_results(results_df)

if __name__ == '__main__':
    main()