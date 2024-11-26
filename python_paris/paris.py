# import numpy as np
# import networkx as nx


# def paris(graph):
#     """
#      Given a graph, compute the paris hierarchy.

#      Parameters
#      ----------
#      dendrogram: networkx.graph
#          A graph with weighted edges.

#      Returns
#      -------
#      dendrogram: numpy.array
#          The paris hierachical clustering is represneted by the dendrogram. Each line of the dendrogram contains the
#          merged nodes, the distance between merged nodes and the number of nodes in the new cluster.

#      References
#      ----------
#      -
#      """
#     graph_copy = graph.copy()
#     nodes = list(graph_copy.nodes())
#     n_nodes = len(nodes)
#     graph_copy = nx.convert_node_labels_to_integers(graph_copy)
#     if nx.get_edge_attributes(graph_copy, 'weight') == {}:
#         for u, v in graph_copy.edges():
#             graph_copy.add_edge(u, v, weight=1)

#     w = {u: 0 for u in range(n_nodes)}
#     wtot = 0
#     for (u, v) in graph_copy.edges():
#         weight = graph_copy[u][v]['weight']
#         w[u] += weight
#         w[v] += weight
#         wtot += 2 * weight
#     s = {u: 1 for u in range(n_nodes)}
#     cc = []
#     dendrogram = []
#     u = n_nodes

#     while n_nodes > 0:
#         chain = [list(graph_copy.nodes())[0]]
#         while chain != []:
#             a = chain.pop()
#             d_min = float("inf")
#             b = -1
#             neighbors_a = list(graph_copy.neighbors(a))
#             for v in neighbors_a:
#                 if v != a:
#                     d = w[v] * w[a] / float(graph_copy[a][v]['weight']) / float(wtot)
#                     if d < d_min:
#                         b = v
#                         d_min = d
#                     elif d == d_min:
#                         b = min(b, v)
#             d = d_min
#             if chain != []:
#                 c = chain.pop()
#                 if b == c:
#                     dendrogram.append([a, b, d, s[a] + s[b]])
#                     graph_copy.add_node(u)
#                     neighbors_a = list(graph_copy.neighbors(a))
#                     neighbors_b = list(graph_copy.neighbors(b))
#                     for v in neighbors_a:
#                         graph_copy.add_edge(u, v, weight=graph_copy[a][v]['weight'])
#                     for v in neighbors_b:
#                         if graph_copy.has_edge(u, v):
#                             graph_copy[u][v]['weight'] += graph_copy[b][v]['weight']
#                         else:
#                             graph_copy.add_edge(u, v, weight=graph_copy[b][v]['weight'])
#                     graph_copy.remove_node(a)
#                     graph_copy.remove_node(b)
#                     n_nodes -= 1
#                     w[u] = w.pop(a) + w.pop(b)
#                     s[u] = s.pop(a) + s.pop(b)
#                     u += 1
#                 else:
#                     chain.append(c)
#                     chain.append(a)
#                     chain.append(b)
#             elif b >= 0:
#                 chain.append(a)
#                 chain.append(b)
#             else:
#                 cc.append((a, s[a]))
#                 graph_copy.remove_node(a)
#                 w.pop(a)
#                 s.pop(a)
#                 n_nodes -= 1

#     a, s = cc.pop()
#     for b, t in cc:
#         s += t
#         dendrogram.append([a, b, float("inf"), s])
#         a = u
#         u += 1

#     return reorder_dendrogram(np.array(dendrogram))


# def reorder_dendrogram(dendrogram):
#     """
#      Given a graph, compute the paris hierarchy

#      Parameters
#      ----------
#      dendrogram: numpy.array
#          Each line of the dendrogram contains the merged nodes, the distance between merged nodes and the number of
#          nodes in the new cluster. The lines are not sorted with respect to increasing distances.

#      Returns
#      -------
#      dendrogram: numpy.array
#          Each line of the dendrogram contains the merged nodes, the distance between merged nodes and the number of
#          nodes in the new cluster. The lines are sorted with respect to increasing distances.

#      References
#      ----------
#      -
#      """
#     n = np.shape(dendrogram)[0] + 1
#     order = np.zeros((2, n - 1), float)
#     order[0] = range(n - 1)
#     order[1] = np.array(dendrogram)[:, 2]
#     index = np.lexsort(order)
#     n_index = {i: i for i in range(n)}
#     n_index.update({n + index[t]: n + t for t in range(n - 1)})
#     return np.array([[n_index[int(dendrogram[t][0])], n_index[int(dendrogram[t][1])], dendrogram[t][2], dendrogram[t][3]] for t in range(n - 1)])[index, :]





import numpy as np
import networkx as nx

def paris(graph):
    """
    Computes the Paris hierarchical clustering for a given graph.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph with weighted edges.

    Returns
    -------
    dendrogram : numpy.array
        Hierarchical clustering represented as a dendrogram. Each row contains:
        [merged node 1, merged node 2, distance, size of new cluster].
    """
    # Step 1: Prepare graph and initialize variables
    graph = preprocess_graph(graph)  # Ensure weights are present and relabel nodes
    n_nodes = len(graph.nodes())  # Number of nodes in the graph
    w, wtot, s = initialize_weights(graph, n_nodes)  # Initialize node weights and cluster sizes
    
    dendrogram = []  # Store the hierarchical clustering steps
    cc = []  # List to hold connected components
    next_cluster_id = n_nodes  # ID for new clusters formed during the process

    # Step 2: Perform hierarchical clustering
    while n_nodes > 0:
        # Start with a chain of nodes for nearest-neighbor clustering
        chain = [list(graph.nodes())[0]]
        while chain:
            a = chain.pop()  # Take the last node in the chain
            b, d = find_closest_neighbor(graph, a, w, wtot)  # Find closest neighbor

            if chain:  # If the chain is not empty
                c = chain.pop()
                if b == c:  # Nodes `a` and `b` can merge
                    dendrogram.append([a, b, d, s[a] + s[b]])  # Record the merge step
                    merge_clusters(graph, a, b, next_cluster_id, w, s)  # Merge clusters
                    next_cluster_id += 1  # Increment cluster ID
                    n_nodes -= 1  # Decrease the node count
                else:  # If no merge happens, extend the chain
                    chain.extend([c, a, b])
            elif b >= 0:  # If a neighbor is found, add it to the chain
                chain.extend([a, b])
            else:  # If no neighbor, isolate the node
                cc.append((a, s[a]))
                graph.remove_node(a)  # Remove isolated node
                w.pop(a)  # Update weights
                s.pop(a)  # Update sizes
                n_nodes -= 1

    # Step 3: Handle any remaining connected components
    process_connected_components(cc, dendrogram, next_cluster_id)

    # Step 4: Reorder the dendrogram rows for standard hierarchical clustering format
    return reorder_dendrogram(np.array(dendrogram))


def preprocess_graph(graph):
    """
    Ensure the graph has weighted edges and relabel nodes for easier processing.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph.

    Returns
    -------
    networkx.Graph
        Processed graph with weights assigned to all edges.
    """
    graph = nx.convert_node_labels_to_integers(graph.copy())  # Relabel nodes as integers
    if not nx.get_edge_attributes(graph, 'weight'):  # If weights are missing
        nx.set_edge_attributes(graph, 1, 'weight')  # Assign default weight of 1
    return graph


def initialize_weights(graph, n_nodes):
    """
    Initialize node weights, total weight of the graph, and cluster sizes.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph.
    n_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    tuple
        (weights, total_weight, sizes), where:
        - weights: dict of node weights
        - total_weight: sum of all edge weights
        - sizes: dict of cluster sizes
    """
    w = {u: 0 for u in range(n_nodes)}  # Node weights
    s = {u: 1 for u in range(n_nodes)}  # Cluster sizes
    wtot = 0  # Total weight of the graph

    # Calculate weights and total weight
    for u, v, data in graph.edges(data=True):
        weight = data['weight']
        w[u] += weight
        w[v] += weight
        wtot += 2 * weight  # Each edge contributes to both nodes

    return w, wtot, s


def find_closest_neighbor(graph, node, weights, total_weight):
    """
    Find the closest neighbor to the given node based on the Paris distance.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph.
    node : int
        Current node.
    weights : dict
        Weights of nodes.
    total_weight : float
        Total weight of all edges in the graph.

    Returns
    -------
    tuple
        (closest_node, min_distance)
    """
    min_distance = float("inf")
    closest_node = -1

    # Iterate through neighbors to find the closest one
    for neighbor in graph.neighbors(node):
        if neighbor != node:
            distance = (weights[node] * weights[neighbor]) / graph[node][neighbor]['weight'] / total_weight
            if distance < min_distance or (distance == min_distance and neighbor < closest_node):
                closest_node, min_distance = neighbor, distance

    return closest_node, min_distance


def merge_clusters(graph, a, b, new_node, weights, sizes):
    """
    Merge two clusters into a new cluster.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph.
    a, b : int
        Nodes to be merged.
    new_node : int
        ID for the new cluster node.
    weights : dict
        Node weights.
    sizes : dict
        Cluster sizes.
    """
    graph.add_node(new_node)  # Add new cluster node

    # Transfer edges from old nodes to the new cluster
    for neighbor in set(graph.neighbors(a)).union(graph.neighbors(b)):
        weight = (graph[a][neighbor]['weight'] if graph.has_edge(a, neighbor) else 0) + \
                 (graph[b][neighbor]['weight'] if graph.has_edge(b, neighbor) else 0)
        graph.add_edge(new_node, neighbor, weight=weight)

    graph.remove_nodes_from([a, b])  # Remove merged nodes
    weights[new_node] = weights.pop(a) + weights.pop(b)  # Update weights
    sizes[new_node] = sizes.pop(a) + sizes.pop(b)  # Update sizes


def process_connected_components(components, dendrogram, next_cluster_id):
    """
    Process remaining connected components and add them to the dendrogram.

    Parameters
    ----------
    components : list
        Remaining connected components.
    dendrogram : list
        Current dendrogram.
    next_cluster_id : int
        Next cluster ID for new clusters.
    """
    a, size = components.pop()
    for b, t in components:
        size += t
        dendrogram.append([a, b, float("inf"), size])
        a = next_cluster_id
        next_cluster_id += 1


def reorder_dendrogram(dendrogram):
    """
    Reorder dendrogram rows by increasing distances.

    Parameters
    ----------
    dendrogram : numpy.array
        Unsorted dendrogram.

    Returns
    -------
    numpy.array
        Sorted dendrogram by increasing distances.
    """
    n = len(dendrogram) + 1
    order = np.lexsort((dendrogram[:, 2], np.arange(len(dendrogram))))  # Sort by distance, then by index
    mapping = {i: i for i in range(n)}  # Initial mapping of indices
    mapping.update({n + order[i]: n + i for i in range(len(dendrogram))})

    # Reorder the dendrogram based on sorted indices
    return np.array([
        [mapping[int(row[0])], mapping[int(row[1])], row[2], row[3]]
        for row in dendrogram[order]
    ])
