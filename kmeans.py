import numpy as np
import random

INF = 9999

def find_radius(graph, clusters):
    # set the initial radius as zero because we want the max
    definitive_radius = 0
    for key, cluster in clusters.items():
        # the max radius will work for the cluster it-self
        max_radius = 0
        for node in cluster:
            # if the distance is greater than the previous radius
            # set the new radius
            max_radius = max(graph[key][node], max_radius)
        # compare the cluster radius with the solution radius
        definitive_radius = max(max_radius, definitive_radius)

    return definitive_radius


def update_centroids(graph, clusters):
    new_centroids = []
    for cluster in clusters.values():
        min_avg_dist = INF
        new_centroid = None

        for node in cluster:
            # define the mean of the node for all other nodes in the cluster
            avg_dist = np.mean([graph[node][other] for other in cluster])

            # if it has the min mean distance set it as it
            if avg_dist < min_avg_dist:
                min_avg_dist = avg_dist
                new_centroid = node
        # for each cluster choose the node with the min mean distance to the other
        # node as the new centroid
        new_centroids.append(new_centroid)

    return new_centroids

def assign_clusters(graph, k, centroids):
    num_nodes = len(graph)
    clusters = {centroid: [] for centroid in centroids}

    for i in range(num_nodes):
        min_dist = INF
        clossest_centroid = None

        # go trought all nodes of the graph and put it in the
        # cluster of the clossest centroid
        for j in range(k):
            if graph[i][centroids[j]] < min_dist:
                min_dist = graph[i][centroids[j]]
                clossest_centroid = centroids[j]
        clusters[clossest_centroid].append(i)

    return clusters

def kmeans(graph, k):
    num_nodes = len(graph)

    # incializa centroids
    centroids = random.sample(range(num_nodes), k)
    # define the clusters for the centroids
    clusters = assign_clusters(graph, k, centroids)
    # find the radius for the centroids
    radius = find_radius(graph, clusters)

    # set the first best cluster as the cluster
    best_clusters = clusters

    while True:
        # change the centroid for the node closest to the center of the cluster
        new_centroids = update_centroids(graph, clusters)

        centroids = new_centroids
        # assing the cluster for the news centroids
        clusters = assign_clusters(graph, k, centroids)
        # define the new radius ass well
        new_radius = find_radius(graph, clusters)

        # stop the kmeans to loop infinity
        if set(new_centroids) == set(centroids):
            # if the solution does not change break
            break
        else:
            # if the soluction is better, set it as the better solution
            best_clusters = clusters
            radius = new_radius
        
    # return the best solution found
    return best_clusters, radius
