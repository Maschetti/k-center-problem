import full

for i in range(1, 11):
    full.run_kmean(f'kmeans{i}.csv')

full.run_brute_force('brute_force.csv')