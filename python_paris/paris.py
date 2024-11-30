import numpy as np
import networkx as nx


def paris(graph):
    """
    Given a graph, compute the Paris hierarchy.

    Parameters
    ----------
    graph: networkx.Graph
        A graph with weighted edges.

    Returns
    -------
    dendrogram: numpy.array
        The Paris hierarchical clustering is represented by the dendrogram. Each line of the dendrogram contains:
        - The merged nodes
        - The distance between merged nodes
        - The number of nodes in the new cluster
    """
    # Preprocess the graph and initialize variables:
    # - `graph_copy`: A copy of the input graph to avoid modifying the original.
    # - `n_nodes`: Total number of nodes in the graph.
    # - `s`: A dictionary storing the size of each cluster (number of nodes in each cluster).
    # - `w`: A dictionary storing weights for each node.
    # - `wtot`: The total weight of the graph.
    graph_copy, n_nodes, s, w, wtot = process_graph(graph)

    # Initialize a list to store connected components and the dendrogram
    cc = []  # Stores nodes that cannot be merged (connected components)
    dendrogram = []  # Stores the hierarchical clustering result

    # Start a unique node index for new clusters
    u = n_nodes

    # Main loop: Continue until all nodes are processed
    while n_nodes > 0:
        # Initialize the "chain" with the first node in the graph
        chain = [list(graph_copy.nodes())[0]]
        # Process the chain until it is empty
        while chain != []:
            # Pop the last node in the chain
            a = chain.pop()
            # Find the closest neighbor (node `b`) and the distance `d`
            b, d = get_closest_neighbor(a, graph_copy, w, wtot)
            # Check if the chain is not empty
            if chain != []:
                # Get the second-to-last node in the chain
                c = chain.pop()
                # If the closest neighbor `b` matches `c`, merge nodes `a` and `b`
                if b == c:
                    # Add the merge operation to the dendrogram
                    dendrogram.append([a, b, d, s[a] + s[b]])
                    # Merge nodes `a` and `b` into a new cluster and update the graph
                    n_nodes = ad_new_cluster(a, b, graph_copy, n_nodes, s, u, w)
                    # Increment the unique cluster index
                    u += 1
                else:
                    # If `b` does not match `c`, push `c`, `a`, and `b` back onto the chain
                    chain.append(c)
                    chain.append(a)
                    chain.append(b)
            elif b >= 0:
                # If chain is empty but a valid neighbor `b` exists, add `a` and `b` back to the chain
                chain.append(a)
                chain.append(b)
            else:
                # If no neighbor exists, add node `a` to the connected components
                cc.append((a, s[a]))
                # Remove node `a` from the graph
                n_nodes = remove_node(a, graph_copy, n_nodes, s, w)

    # Handle remaining connected components in `cc` (node that are still alone)
    process_remaining_components(cc, dendrogram, u)

    # Reorder the dendrogram and return it as a numpy array
    return reorder_dendrogram(np.array(dendrogram))


def process_remaining_components(cc, dendrogram, u):
    # Start with the first remaining node
    a, s = cc.pop()
    for b, t in cc:
        # Update the cluster size `s` and add the merge operation to the dendrogram
        s += t
        dendrogram.append([a, b, float("inf"), s])
        # Increment the unique cluster index
        a = u
        u += 1


def remove_node(a, graph_copy, n_nodes, s, w):
    graph_copy.remove_node(a)
    w.pop(a)
    s.pop(a)
    n_nodes -= 1
    return n_nodes


def ad_new_cluster(a, b, graph_copy, n_nodes, s, u, w):
    """
    Merge two nodes (a and b) into a new cluster (u) and update the graph.
    """
    # Add a new node `u` to represent the merged cluster
    graph_copy.add_node(u)

    # Get the neighbors of node `a` and node `b`
    neighbors_a = list(graph_copy.neighbors(a))
    neighbors_b = list(graph_copy.neighbors(b))

    # Add edges from the new cluster `u` to the neighbors of `a`
    for v in neighbors_a:
        graph_copy.add_edge(u, v, weight=graph_copy[a][v]['weight'])

    # Add edges from the new cluster `u` to the neighbors of `b`
    for v in neighbors_b:
        if graph_copy.has_edge(u, v):
            # If an edge already exists between `u` and `v`, combine the weights
            graph_copy[u][v]['weight'] += graph_copy[b][v]['weight']
        else:
            # Otherwise, add a new edge between `u` and `v`
            graph_copy.add_edge(u, v, weight=graph_copy[b][v]['weight'])

    # Remove the original nodes `a` and `b` from the graph
    graph_copy.remove_node(a)
    graph_copy.remove_node(b)

    # Decrease the total number of nodes
    n_nodes -= 1

    # Update the weight of the new cluster `u` as the sum of weights of `a` and `b`
    w[u] = w.pop(a) + w.pop(b)

    # Update the size of the new cluster `u` as the sum of sizes of `a` and `b`
    s[u] = s.pop(a) + s.pop(b)

    # Return the updated number of nodes
    return n_nodes



def get_closest_neighbor(a, graph_copy, w, wtot):
    d_min = float("inf")
    b = -1
    neighbors_a = list(graph_copy.neighbors(a))
    for v in neighbors_a:
        if v != a:
            d = w[v] * w[a] / float(graph_copy[a][v]['weight']) / float(wtot)
            if d < d_min:
                b = v
                d_min = d
            elif d == d_min:
                b = min(b, v)
    d = d_min
    return b, d


def process_graph(graph):
    graph_copy = graph.copy()
    nodes = list(graph_copy.nodes())
    n_nodes = len(nodes)
    graph_copy = nx.convert_node_labels_to_integers(graph_copy)
    if nx.get_edge_attributes(graph_copy, 'weight') == {}:
        for u, v in graph_copy.edges():
            graph_copy.add_edge(u, v, weight=1)
    w = {u: 0 for u in range(n_nodes)}
    wtot = 0
    for (u, v) in graph_copy.edges():
        weight = graph_copy[u][v]['weight']
        w[u] += weight
        w[v] += weight
        wtot += 2 * weight
    s = {u: 1 for u in range(n_nodes)}
    return graph_copy, n_nodes, s, w, wtot


def reorder_dendrogram(dendrogram):
    """
     Given a graph, compute the paris hierarchy

     Parameters
     ----------
     dendrogram: numpy.array
         Each line of the dendrogram contains the merged nodes, the distance between merged nodes and the number of
         nodes in the new cluster. The lines are not sorted with respect to increasing distances.

     Returns
     -------
     dendrogram: numpy.array
         Each line of the dendrogram contains the merged nodes, the distance between merged nodes and the number of
         nodes in the new cluster. The lines are sorted with respect to increasing distances.

     References
     ----------
     -
     """
    n = np.shape(dendrogram)[0] + 1
    order = np.zeros((2, n - 1), float)
    order[0] = range(n - 1)
    order[1] = np.array(dendrogram)[:, 2]
    index = np.lexsort(order)
    n_index = {i: i for i in range(n)}
    n_index.update({n + index[t]: n + t for t in range(n - 1)})
    return np.array([[n_index[int(dendrogram[t][0])], n_index[int(dendrogram[t][1])], dendrogram[t][2], dendrogram[t][3]] for t in range(n - 1)])[index, :]