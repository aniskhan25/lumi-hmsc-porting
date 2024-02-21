import sys
import numpy as np
import torch
import timeit

# Running on LUMI with pytorch container module loaded
# ml use /appl/local/csc/modulefiles
# module load pytorch
# python3 cholesky_torch_benchmark.py 10,100,1000,10000,15000,20000,25000 10 double
#
# Puhti:
# srun -p gputest --nodes=1 --ntasks-per-node=1 --mem=32G --gres=gpu:v100:1 -t 0:15:00 python3 cholesky_torch_benchmark.py 10,100,1000,10000,15000,20000,25000 10 double
#
# Mahti:
# srun -p gputest --nodes=1 --ntasks-per-node=1 --mem=32G --gres=gpu:a100:1 -t 0:15:00 python3 cholesky_torch_benchmark.py 10,100,1000,10000,15000,20000,25000 10 double

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
        average = timeit.timeit(run, number=M) / M
        print("Avg. time per run for size {:8}: {:.5e} sec".format(N, average), flush=True)
        averages[N] = average
    print(averages)
