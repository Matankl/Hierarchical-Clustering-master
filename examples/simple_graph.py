# -*- coding: utf-8 -*-
"""
=============================================
Demo of the Paris algorithm on a simple graph
=============================================
"""
import math

print(__doc__)

from community import best_partition
from python_paris.paris import *
from python_paris.cluster_cut_slicer import *
from python_paris.homogeneous_cut_slicer import *
from python_paris.heterogeneous_cut_slicer import *
from python_paris.distance_slicer import *

import random


def generate_graph_with_euclidean_weights(num_nodes, additional_edges=0):
    """
    Generate a graph with nodes and edges where weights are based on Euclidean distances.

    Parameters:
        num_nodes (int): The number of nodes in the graph.
        additional_edges (int): The number of extra edges to add randomly beyond the minimum.

    Returns:
        tuple: A list of edges (u, v, weight) and a dictionary of node positions.
    """
    # Generate random 2D positions for each node
    node_positions = {i: (random.uniform(0, 100), random.uniform(0, 100)) for i in range(num_nodes)}

    # Ensure all nodes are connected by forming a linear chain
    edges = []
    for i in range(num_nodes - 1):
        u, v = i, i + 1
        pos_u, pos_v = node_positions[u], node_positions[v]
        weight = math.sqrt((pos_u[0] - pos_v[0]) ** 2 + (pos_u[1] - pos_v[1]) ** 2)
        edges.append((u, v, weight))

    # Add additional random edges
    existing_edges = set((u, v) for u, v, _ in edges)
    for _ in range(additional_edges):
        while True:
            u = random.randint(0, num_nodes - 1)
            v = random.randint(0, num_nodes - 1)
            if u != v and (u, v) not in existing_edges and (v, u) not in existing_edges:
                pos_u, pos_v = node_positions[u], node_positions[v]
                weight = math.sqrt((pos_u[0] - pos_v[0]) ** 2 + (pos_u[1] - pos_v[1]) ** 2)
                edges.append((u, v, weight))
                existing_edges.add((u, v))
                break

    return edges, node_positions


# ############################################################################################
# Generate the graph
N_NODES = 30
ADDITIONAL_EDGES = int(N_NODES * 0.4)
graph = nx.Graph()
graph.add_nodes_from([i for i in range(0, N_NODES)])
edges, positions = generate_graph_with_euclidean_weights(N_NODES, ADDITIONAL_EDGES)
graph.add_weighted_edges_from(edges)


# ############################################################################################
# Apply Paris on the graph
print("Apply the algorithm to the NetworkX Graph object")
dendrogram = paris(graph)

# ############################################################################################
# Process dendrogram
best_cut, best_score = best_cluster_cut(dendrogram)
best_cluster = clustering_from_cluster_cut(dendrogram, best_cut)

best_cut, best_score = best_homogeneous_cut(dendrogram)
best_homogeneous_clustering = clustering_from_homogeneous_cut(dendrogram, best_cut)

best_cut, best_score = best_heterogeneous_cut(dendrogram)
best_heterogeneous_clustering = clustering_from_heterogeneous_cut(dendrogram, best_cut)

best_dist, best_score = best_distance(dendrogram)
best_louvain_clustering = best_partition(graph, resolution=best_dist)

# #############################################################################
# Plot result
print("Plot the result\n")
import matplotlib.pyplot as plt

colors = ['b', 'g', 'r', 'c', 'm', 'y']
pos = nx.fruchterman_reingold_layout(graph, pos=positions, iterations=500, dim = 2)

plt.plot()
plt.title('Best cluster')
plt.axis('off')
nx.draw_networkx_edges(graph, pos)
nodes = nx.draw_networkx_nodes(graph, pos, node_color='k')
nx.draw_networkx_nodes(graph, pos, nodelist=best_cluster, node_color=colors[0])
plt.show()
