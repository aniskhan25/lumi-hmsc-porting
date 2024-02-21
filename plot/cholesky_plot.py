import numpy as np
import matplotlib.pyplot as plt

matrix_sizes = []

# Plot the line graph
plt.figure(figsize=(10, 6))
for lib in ['tensorflow', 'pytorch']:
    for hpc in ['lumi', 'mahti', 'puhti']:
        fpath = f'../data/cholesky_{lib}_{hpc}.dat'
        data = np.loadtxt(fpath)
        ls = {'tensorflow': '-', 'pytorch': '--'}[lib]
        gpu = {'lumi': 'MI250X / 1 GCD', 'mahti': 'A100', 'puhti': 'V100'}[hpc]
        plt.plot(data[:, 0], data[:, 1], marker='o', ls=ls, label=f'{lib} {hpc} ({gpu})')
        matrix_sizes.append(data[:, 0].astype(int))

matrix_sizes = np.unique(matrix_sizes)

plt.xscale('log')
plt.yscale('log')
plt.title('Time vs Matrix Size')
plt.xlabel('Matrix Size')
plt.ylabel('Time (seconds)')
plt.legend()
plt.grid(True)
plt.xticks(matrix_sizes, [str(size) for size in matrix_sizes], rotation=45)
plt.subplots_adjust()

plt.savefig('cholesky_plot.png')
plt.show()
