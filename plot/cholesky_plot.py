import matplotlib.pyplot as plt

# Data for PyTorch Cholesky decomposition
data_pytorch = {
    10: 0.00010881200432777405,
    100: 0.00016832035034894944,
    1000: 0.004341504909098149,
    10000: 0.11293962243944407,
    15000: 0.22000754941254855,
    20000: 0.37668313439935447,
    25000: 0.6280569981783628
}

# Data for TensorFlow Cholesky decomposition
data_tensorflow = {
    10: 0.00029554292559623717,
    100: 0.0011621629819273949,
    1000: 0.010687164403498173,
    10000: 0.2030246179550886,
    15000: 0.4577014818787575,
    20000: 0.8616737483069301,
    25000: 1.5558607606217265
}

matrix_sizes_pytorch = list(data_pytorch.keys())
times_pytorch = list(data_pytorch.values())

matrix_sizes_tensorflow = list(data_tensorflow.keys())
times_tensorflow = list(data_tensorflow.values())

# Plot the line graph
plt.figure(figsize=(10, 6))
plt.plot(matrix_sizes_pytorch, times_pytorch, marker='o', color='b', linestyle='-', label='PyTorch Cholesky Lumi')
plt.plot(matrix_sizes_tensorflow, times_tensorflow, marker='o', color='r', linestyle='-', label='TensorFlow Cholesky Lumi')

plt.xscale('log') 
plt.yscale('log') 
plt.title('Time vs Matrix Size')
plt.xlabel('Matrix Size')
plt.ylabel('Time (seconds)')
plt.legend()
plt.grid(True)
plt.xticks(matrix_sizes_pytorch + matrix_sizes_tensorflow, [str(size) for size in matrix_sizes_pytorch] + [str(size) for size in matrix_sizes_tensorflow], rotation=45)

plt.savefig('cholesky_plot.png')
plt.show()