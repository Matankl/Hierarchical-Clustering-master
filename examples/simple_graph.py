# -*- coding: utf-8 -*-
"""
=============================================
Demo of the Paris algorithm on a simple graph
=============================================
"""
print(__doc__)

from community import best_partition
from python_paris.paris import *
from python_paris.cluster_cut_slicer import *
from python_paris.homogeneous_cut_slicer import *
from python_paris.heterogeneous_cut_slicer import *
from python_paris.distance_slicer import *

# ############################################################################################
# Generate the graph
graph = nx.Graph()
graph.add_nodes_from([0, 1, 2, 3, 4, 5, 6 ,7, 8, 9, 10, 11, 12, 13, 14])
graph.add_weighted_edges_from([(0, 1, 1), (0, 2, 1), (1, 2, 1), (2, 3, 1),
                      (3, 4, 1), (3, 5, 1), (4, 5, 1), (5, 6, 1), (5, 7, 1), (6, 7, 1), (7, 8, 1), (7, 9, 1), (8, 9, 1), (9, 10, 1), (9, 11, 1), (10, 11, 1), (11, 12, 1), (11, 13, 1), (12, 13, 1), (13, 14, 1)])


# import networkx as nx
# import random

# # Create a new graph
# complex_graph = nx.Graph()

# # Add nodes
# num_nodes = 15  # Adjust the number of nodes for complexity
# complex_graph.add_nodes_from(range(num_nodes))

# # Add random edges
# num_edges = 0  # Adjust the number of edges for complexity
# for _ in range(num_edges):
#     u, v = random.sample(range(num_nodes), 2)  # Randomly select two distinct nodes
#     weight = random.randint(1, 10)  # Assign a random weight to the edge
#     complex_graph.add_edge(u, v, weight=weight)

# # Add some cliques (fully connected subgraphs)
# clique_nodes = [random.sample(range(num_nodes), 5) for _ in range(3)]  # Create 3 cliques of size 5
# for clique in clique_nodes:
#     complex_graph.add_edges_from([(clique[i], clique[j]) for i in range(len(clique)) for j in range(i + 1, len(clique))])

# # Add a star structure (one central node connected to many others)
# central_node = random.choice(range(num_nodes))
# leaf_nodes = random.sample(range(num_nodes), 8)  # Select 8 nodes for the star
# complex_graph.add_edges_from([(central_node, leaf) for leaf in leaf_nodes])

# # Add a line structure (a path connecting a sequence of nodes)
# line_nodes = random.sample(range(num_nodes), 10)  # Select 10 nodes for the line
# for i in range(len(line_nodes) - 1):
#     complex_graph.add_edge(line_nodes[i], line_nodes[i + 1])

# # Add a cycle
# cycle_nodes = random.sample(range(num_nodes), 6)  # Select 6 nodes for the cycle
# for i in range(len(cycle_nodes)):
#     complex_graph.add_edge(cycle_nodes[i], cycle_nodes[(i + 1) % len(cycle_nodes)])

# #make sure all edges have a weight
# graph = complex_graph
# for u, v in graph.edges:
#     if 'weight' not in graph[u][v]:
#         graph[u][v]['weight'] = 1  # Assign a default weight


# print("Complex graph created with:")
# print(f"- {complex_graph.number_of_nodes()} nodes")
# print(f"- {complex_graph.number_of_edges()} edges")


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
pos = nx.fruchterman_reingold_layout(graph)

plt.subplot(2, 2, 1)
plt.title('Best cluster')
plt.axis('off')
nx.draw_networkx_edges(graph, pos)
nodes = nx.draw_networkx_nodes(graph, pos, node_color='k')
nx.draw_networkx_nodes(graph, pos, nodelist=best_cluster, node_color=colors[0])

plt.subplot(2, 2, 2)
plt.title('Best homogeneous clustering')
plt.axis('off')
nx.draw_networkx_edges(graph, pos)
for l in range(min(len(colors), len(best_homogeneous_clustering))):
    nx.draw_networkx_nodes(graph, pos, nodelist=best_homogeneous_clustering[l], node_color=colors[l])

plt.subplot(2, 2, 3)
plt.title('Best heterogeneous clustering')
plt.axis('off')
nx.draw_networkx_edges(graph, pos)
for l in range(min(len(colors), len(best_heterogeneous_clustering))):
    nx.draw_networkx_nodes(graph, pos, nodelist=best_heterogeneous_clustering[l], node_color=colors[l])

plt.subplot(2, 2, 4)
plt.title('Best distance')
plt.axis('off')
nx.draw_networkx_edges(graph, pos)
nx.draw_networkx_nodes(graph, pos, node_color=[colors[best_louvain_clustering[node]] for node in graph])
plt.show()
