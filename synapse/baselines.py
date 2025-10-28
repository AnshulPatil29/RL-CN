import networkx as nx

def ospf_routing(graph, source, destination, weight='weight'):
    """
    Calculates the shortest path using Dijkstra's algorithm, simulating OSPF.
    
    Args:
        graph (nx.Graph): The network graph.
        source (int): The starting node.
        destination (int): The target node.
        weight (str): The edge attribute to use as cost. Defaults to 'weight'.

    Returns:
        list: The list of nodes forming the shortest path.
              Returns an empty list if no path exists.
    """
    try:
        path = nx.shortest_path(graph, source=source, target=destination, weight=weight)
        return path
    except nx.NetworkXNoPath:
        return []

if __name__ == '__main__':
    # --- Test Script ---
    print("Testing OSPF (Dijkstra's) baseline...")

    # Create a simple graph for testing
    G = nx.Graph()
    edges = [(0, 1, 10), (0, 2, 20), (1, 2, 5), (1, 3, 15), (2, 3, 5)]
    # Add edges with weights
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    source_node = 0
    dest_node = 3

    # Calculate shortest path
    path = ospf_routing(G, source_node, dest_node)
    
    print(f"Graph nodes: {list(G.nodes())}")
    print(f"Graph edges (with weights): {G.edges(data=True)}")
    print(f"\nCalculating shortest path from {source_node} to {dest_node}...")
    
    if path:
        print(f"Shortest path found: {path}")
        # Verify the path
        # Path 0->1->2->3 has cost 10+5+5=20
        # Path 0->2->3 has cost 20+5=25
        # Path 0->1->3 has cost 10+15=25
        expected_path = [0, 1, 2, 3]
        assert path == expected_path, f"Test failed! Expected {expected_path}, but got {path}"
        print("Path is correct. Test PASSED.")
    else:
        print("No path found. Test FAILED.")

    # Test a case with no path
    G.add_node(4) # Add an isolated node
    print(f"\nCalculating shortest path from {source_node} to 4 (isolated node)...")
    no_path = ospf_routing(G, source_node, 4)
    if not no_path:
        print("Correctly returned no path. Test PASSED.")
    else:
        print(f"Found an unexpected path: {no_path}. Test FAILED.")