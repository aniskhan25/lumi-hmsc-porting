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
# python3 cholesky_benchmark.py 10,100,1000,10000,15000,20000,25000 10 double

def chol_tf(A, dtype=tf.float64):
    with tf.device('/GPU:0'):
        Atf = tf.Variable(A, dtype=dtype)
    def run():
        with tf.device('/GPU:0'):
            tf.linalg.cholesky(Atf)
    return run

if __name__ == '__main__':
    matrix_sizes, times, precision = sys.argv[1:]
    ms_str = matrix_sizes.split(',')
    averages = {}

    # Set precision
    dtype = tf.float64 if precision == 'double' else tf.float32

    for size in ms_str:
        N = int(size)
        M = int(times)
        L = np.random.randn(N, N)
        L[np.triu_indices(N, k=1)] = 0.
        A = np.dot(L, L.T)
        jitter = 1e-6
        A[np.diag_indices(N)] += jitter
        A_tensor = tf.convert_to_tensor(A, dtype=dtype)
        run = chol_tf(A_tensor, dtype=dtype)
        # Warmup
        run()
        # Timed choleskys
        averages[N] = timeit.timeit(run, number=M) / M
        print("Avg. time per run for size {}: {:.5e}sec".format(N, timeit.timeit(run, number=M) / M))

