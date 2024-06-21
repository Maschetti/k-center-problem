from itertools import combinations
from kmeans import find_radius, assign_clusters

INF = 9999

def brute_force(graph, k):
    nodes = [i for i in range(len(graph))]

    # set all subgroups of size k, using the all nodes of the graph
    sub_groups = list(combinations(nodes, k))
    # the best radius is the smallest, so set it as INF
    best_radius = INF
    for sub_group in sub_groups:
        # set the clusters for each subgroup
        clusters = assign_clusters(graph, k, sub_group)
        # set the radius for reach subgroup
        radius = find_radius(graph, clusters)

        # define the best solution as the smallest radius
        if radius < best_radius:
            best_clusters = clusters
            best_radius = radius
    
    return best_clusters, best_radius
