import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def plot_network_graph(graph, ax, title="Network Topology"):
    """
    Plots the network graph with node labels.
    """
    pos = nx.spring_layout(graph, seed=42)
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=10, font_weight='bold', ax=ax)
    ax.set_title(title)

def plot_traffic_matrix(matrix, ax, title="Traffic Matrix"):
    """
    Plots a heatmap of the traffic matrix.
    """
    im = ax.imshow(matrix, cmap='viridis', interpolation='nearest')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Destination Node")
    ax.set_ylabel("Source Node")

def plot_metrics(metrics_history, y_label, title, ax):
    """
    Plots a metric over time (episodes).
    """
    ax.plot(metrics_history)
    ax.set_xlabel("Episode")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True)

if __name__ == '__main__':
    # --- Test Script ---
    print("Testing utility functions...")

    # Create a sample graph for testing 
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])

    # Create a sample traffic matrix
    traffic = np.random.rand(3, 3)
    
    # Create a sample metrics history
    history = [100, 95, 92, 88, 85]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Test plotting functions
    plot_network_graph(G, axes[0], "Test Graph")
    print("Graph plot generated.")
    
    plot_traffic_matrix(traffic, axes[1], "Test Traffic")
    print("Traffic matrix plot generated.")

    plot_metrics(history, "Latency (ms)", "Test Latency Over Time", axes[2])
    print("Metrics plot generated.")

    plt.tight_layout()
    plt.show()
    
    print("\nUtility functions test complete. Check the plots.")