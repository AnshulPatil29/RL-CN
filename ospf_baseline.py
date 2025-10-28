
import networkx as nx
from network_env import NetworkEnv
import config

def evaluate_ospf(env, num_steps=100):
    total_max_util = 0
    total_avg_latency = 0

    print("\n--- Evaluating OSPF Baseline ---")

    nx.set_edge_attributes(env.graph, 1, 'weight')

    for i in range(num_steps):
        env._generate_traffic_matrix()

        nx.set_edge_attributes(env.graph, 0.0, 'utilization')
        total_latency_step = 0
        total_traffic_step = 0

        for (src, dst), demand in env.traffic_matrix.items():
            try:
                path = nx.shortest_path(env.graph, source=src, target=dst, weight='weight')
                path_latency = len(path) - 1
                total_latency_step += path_latency * demand
                total_traffic_step += demand

                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    env.graph[u][v]['utilization'] += demand
            except nx.NetworkXNoPath:
                continue

        utilizations = [attr['utilization'] / attr['capacity'] for _, _, attr in env.graph.edges(data=True)]
        total_max_util += max(utilizations) if utilizations else 0
        total_avg_latency += total_latency_step / total_traffic_step if total_traffic_step > 0 else 0

    avg_max_util = total_max_util / num_steps
    avg_latency = total_avg_latency / num_steps

    print(f"OSPF Avg Max Link Utilization: {avg_max_util:.4f}")
    print(f"OSPF Avg Path Latency: {avg_latency:.4f}")

    return avg_max_util, avg_latency

if __name__ == '__main__':
    env = NetworkEnv(config.TOPOLOGY_FILE, num_agents=1)
    evaluate_ospf(env, num_steps=config.MAX_STEPS_PER_EPISODE * config.NUM_EPISODES)
