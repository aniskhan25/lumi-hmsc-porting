import sys
import numpy as np
import tensorflow as tf
import timeit

# Usage:
# python cholesky_benchmark.py MATRIX_SIZES_AS_CSV N_REPEATS
#
# E.g. running on LUMI:
# ml use /appl/local/csc/modulefiles
# ml tensorflow/2.12
# srun -p dev-g --nodes=1 --ntasks-per-node=1 --mem=32G --gpus-per-node=1 -t 00:15:00 python3 cholesky_benchmark.py 10,100,1000,10000,15000,20000,25000 10 double
#
# Puhti:
# srun -p gputest --nodes=1 --ntasks-per-node=1 --mem=32G --gres=gpu:v100:1 -t 0:15:00 python3 cholesky_benchmark.py 10,100,1000,10000,15000,20000,25000 10 double
#
# Mahti:
# srun -p gputest --nodes=1 --ntasks-per-node=1 --mem=32G --gres=gpu:a100:1 -t 0:15:00 python3 cholesky_benchmark.py 10,100,1000,10000,15000,20000,25000 10 double

def chol_tf(A, dtype=tf.double):
    n = [1,2]
    n = tf.convert_to_tensor(n, dtype=tf.float64)
    with tf.device('/GPU:0'):
        Atf = tf.Variable(A, dtype=dtype)
    def run():
        with tf.device('/GPU:0'):
            return cholesky(A,n)
    cholesky(A,n)
    return run

np.set_printoptions(precision=3, suppress=True, linewidth=200, floatmode='fixed')
N_MAX_PRINT = 13;

if __name__ == '__main__':
    cholesky_lib=tf.load_op_library('./cuda_op_kernel.so')
    cholesky=cholesky_lib.example
    matrix_sizes, times, precision = sys.argv[1:]
    do_random = False
    matrix_sizes = map(int, matrix_sizes.split(','))
    M = int(times)
    averages = {}

    # Set precision
    dtype = tf.float64 if precision == 'double' else tf.float32

    for N in matrix_sizes:
        if do_random:
            L = np.random.randn(N, N)
            L[np.triu_indices(N, k=1)] = 0.
            A = np.dot(L, L.T)
            jitter = 1e-6
            A[np.diag_indices(N)] += jitter
        else:
            A = np.arange(N, dtype=float)[:, None] + np.arange(N, dtype=float)[None, :]
            A /= N*N
            np.fill_diagonal(A, np.arange(N) + 1)

        if N < N_MAX_PRINT:
            print("Input matrix")
            print(A)

        A_tensor = tf.convert_to_tensor(A, dtype=dtype)
        run = chol_tf(A_tensor, dtype=dtype)

        # Warmup
        L = run()

        if N < N_MAX_PRINT:
            print("Output matrix")
            print(L.numpy())

        # Timed choleskys
        average = timeit.timeit(run, number=M) / M
        print("Avg. time per run for size {:8}: {:.5e} sec".format(N, average), flush=True)
        averages[N] = average
    print(averages)

