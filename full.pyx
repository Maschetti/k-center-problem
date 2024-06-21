from pathlib import Path
import numpy as np
import random
from itertools import combinations
import pandas as pd
import time
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


def build_matrix(edges, num_nodes):
    graph = np.full((num_nodes, num_nodes), INF)

    # insert the values inside the matrix, edge by edge
    for e in edges:
        is_inf = graph[e[0] - 1][e[1] - 1] == INF
        if is_inf:
            graph[e[0] - 1][e[1] - 1] = e[2]
            graph[e[1] - 1][e[0] - 1] = e[2]
            # ensure symetri
        else:
            graph[e[0] - 1][e[1] - 1] += e[2]
            graph[e[1] - 1][e[0] - 1] += e[2]


    # put zeros in the main diagonal
    for i in range(num_nodes):
        graph[i][i] = 0

    return graph

def floyd_warshall(graph):
    number_nodes = len(graph)

    for k in range(number_nodes):
        for i in range(number_nodes):
            for j in range(number_nodes):
                graph[i][j] = min(graph[i][j],graph[i][k] + graph[k][j])

def read_graph_file(file_path):
    with Path.open(file_path, 'r') as file:
        lines = file.readlines()

    num_nodes, num_edges, k = map(int, lines[0].strip().split())

    edges = []

    for line in lines[1:]:
        origin, destiny, weight = map(int, line.strip().split())
        edges.append([origin, destiny, weight])

    return num_nodes, num_edges, k, edges

def run_kmean(csv_file):
    list_files = [i for i in range(1, 41)]
    for i in list_files:
        print(i)
        num_nodes, num_edges, k, edges = read_graph_file(f'entradas\pmed{i}.txt')
        graph = build_matrix(edges, num_nodes)
        floyd_warshall(graph)

        # execute kmeans
        start = time.time()
        clusters, radius = kmeans(graph, k)
        # clusters, radius = brute_force(graph, k)
        end = time.time()

        duration = end - start

        df = pd.read_csv(csv_file)

        df.loc[len(df)] = {'Radius': radius, 'Time': duration}

        df.to_csv(csv_file, index=False)

def run_brute_force(csv_file):
    list_files = [i for i in range(1, 41)]
    for i in list_files:
        print("errou")
        print(i)
        num_nodes, num_edges, k, edges = read_graph_file(f'entradas\pmed{i}.txt')
        graph = build_matrix(edges, num_nodes)
        floyd_warshall(graph)

        # execute kmeans
        start = time.time()
        # clusters, radius = kmeans(graph, k)
        clusters, radius = brute_force(graph, k)
        end = time.time()

        duration = end - start

        df = pd.read_csv(csv_file)

        df.loc[len(df)] = {'Radius': radius, 'Time': duration}

        df.to_csv(csv_file, index=False)