import sys
import numpy as np
import tensorflow as tf
import timeit

# Usage:
# python cholesky_benchmark.py MATRIX_SIZES_AS_CSV N_REPEATS
#
# E.g. running on LUMI:
# ml use /appl/local/csc/modulefiles^C
# ml tensorflow/2.12
# python3 cholesky_benchmark.py 10,100,10000,15000,20000,25000 10


def chol_tf(A):
    with tf.device('/GPU:0'):
        Atf = tf.Variable(A)
    def run():
        with tf.device('/GPU:0'):
            tf.linalg.cholesky(Atf)
    return run


if __name__ == '__main__':
    matrix_sizes, times = sys.argv[1:]
    ms_str = matrix_sizes.split(',')
    averages={}
    for size in ms_str:
        tf.debugging.set_log_device_placement(True)
        N = int(size)
        M = int(times)
        L = np.random.randn(N, N)
        L[np.triu_indices(N, k=1)] = 0.
        A = L @ L.T
        jitter = 1e-6
        A[np.diag_indices(N)] += jitter
        run = chol_tf(A)
        # Warmup data movement
        run()
        # Timed choleskys
        averages[N] = timeit.timeit(run, number=M) / M;
        print("Avg. time per run for size {}: {:.5e}sec".format(N,timeit.timeit(run, number=M) / M))
    print(averages)