from read_file import read_graph_file
from floyd_warshall import build_matrix, floyd_warshall
from kmeans import kmeans
from brute_force import brute_force
import time
import pandas as pd
from pathlib import Path

if __name__ == '__main__':
    list_files = [i for i in range(1, 41)]
    # data = {
    #     'Time': [],
    #     'Radius': [],
    # }
    script_dir = Path(__file__).resolve().parent
    entradas_dir = script_dir / 'entradas'
    file_csv = f'{script_dir}\\brute_force.csv'
    for i in list_files:
        print(i)
        num_nodes, num_edges, k, edges = read_graph_file(f'{entradas_dir}\pmed{i}.txt')
        graph = build_matrix(edges, num_nodes)
        floyd_warshall(graph)

        # execute kmeans
        start = time.time()
        # clusters, radius = kmeans(graph, k)
        clusters, radius = brute_force(graph, k)
        end = time.time()

        duration = end - start

        df = pd.read_csv(file_csv)

        df.loc[len(df)] = {'Radius': radius, 'Time': duration}

        df.to_csv(file_csv, index=False)
