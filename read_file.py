from pathlib import Path

def read_graph_file(file_path):
    with Path.open(file_path, 'r') as file:
        lines = file.readlines()

    num_nodes, num_edges, k = map(int, lines[0].strip().split())

    edges = []

    for line in lines[1:]:
        origin, destiny, weight = map(int, line.strip().split())
        edges.append([origin, destiny, weight])

    return num_nodes, num_edges, k, edges

if __name__ == '__main__':
    num_nodes, num_edges, k, edges = read_graph_file('src\entradas\pmed1.txt')

    print("Number of nodes:", num_nodes)
    print("Number of edges:", num_edges)
    print("k:", k)
    print("Edges:")
    for edge in edges:
        print(edge)
