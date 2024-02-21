import sys
import numpy as np
import torch
import timeit

def chol_pytorch(A_tensor):
    def run():
        torch.linalg.cholesky(A_tensor)
    return run

if __name__ == '__main__':
    matrix_sizes, times, precision = sys.argv[1:]
    ms_str = matrix_sizes.split(',')
    averages = {}
    
    # Set precision
    dtype = torch.float64 if precision == 'double' else torch.float32

    # Move device initialization outside the loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for size in ms_str:
        N = int(size)
        M = int(times)
        L = np.random.randn(N, N)
        L[np.triu_indices(N, k=1)] = 0.
        A = np.dot(L, L.T)
        jitter = 1e-6
        A[np.diag_indices(N)] += jitter
        A_tensor = torch.tensor(A, dtype=dtype, device=device)  # Move A_tensor assignment outside run function
        run = chol_pytorch(A_tensor)
        # Warmup
        run()
        # Timed choleskys
        averages[N] = timeit.timeit(run, number=M) / M
        print("Avg. time per run for size {}: {:.5e}sec".format(N, timeit.timeit(run, number=M) / M))
    print(averages)