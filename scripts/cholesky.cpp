/*
 * This code is based on the CUDA Library Samples by NVIDIA Corporation
 *   https://github.com/NVIDIA/CUDALibrarySamples/
 * licensed under (BSD-3-Clause):
 *
 * Copyright (c) 2022 NVIDIA CORPORATION AND AFFILIATES.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <list>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifdef CUDA

#include <cuda_runtime.h>
#include <cusolverDn.h>

#else

#include <hip/hip_runtime.h>
#include "rocblas/rocblas.h"
#include "rocsolver/rocsolver.h"

#define cudaError_t                           hipError_t
#define cudaSuccess                           hipSuccess
#define cudaMalloc                            hipMalloc
#define cudaFree                              hipFree
#define cudaDeviceReset                       hipDeviceReset
#define cudaDeviceSynchronize                 hipDeviceSynchronize

#define cudaMemcpyAsync                       hipMemcpyAsync
#define cudaMemcpy                            hipMemcpy
#define cudaMemcpyDeviceToDevice              hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost                hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice                hipMemcpyHostToDevice

#define cudaStream_t                          hipStream_t
#define cudaStreamDestroy                     hipStreamDestroy
#define cudaStreamSynchronize                 hipStreamSynchronize
#define cudaStreamNonBlocking                 hipStreamNonBlocking
#define cudaStreamCreateWithFlags             hipStreamCreateWithFlags

#define cusolverStatus_t                      rocblas_status
#define CUSOLVER_STATUS_SUCCESS               rocblas_status_success
#define cublasFillMode_t                      rocblas_fill
#define CUBLAS_FILL_MODE_LOWER                rocblas_fill_lower
#define CUBLAS_FILL_MODE_UPPER                rocblas_fill_upper
#define cusolverDnHandle_t                    rocblas_handle
#define cusolverDnCreate                      rocblas_create_handle
#define cusolverDnDestroy                     rocblas_destroy_handle
#define cusolverDnSetStream                   rocblas_set_stream

#endif


// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                          \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

// cusolver API error checking
#define CUSOLVER_CHECK(err)                                                                        \
    do {                                                                                           \
        cusolverStatus_t err_ = (err);                                                             \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
            throw std::runtime_error("cusolver error");                                            \
        }                                                                                          \
    } while (0)


constexpr int N_MAX_PRINT = 3;


template <typename T>
void print_matrix(const int &n, const std::vector<T> &A) {
    // Print transpose
    for (int i = 0; i < n; i++) {
        if (N_MAX_PRINT < i && i < n - N_MAX_PRINT - 1) {
            if (i == N_MAX_PRINT + 1) {
                for (int j = 0; j < (N_MAX_PRINT + 1) * 2 + 1; j++) {
                    std::printf(" %14s", "...");
                }
                std::cout << "\n";
            }
            continue;
        }
        for (int j = 0; j < n; j++) {
            if (N_MAX_PRINT < j && j < n - N_MAX_PRINT - 1) {
                if (j == N_MAX_PRINT + 1) {
                    std::printf(" %14s", "...");
                }
                continue;
            }
            std::printf(" %14.6e", A[j * n + i]);
        }
        std::cout << "\n";
    }
    std::cout << std::flush;
}

#ifdef CUDA
template <typename T>
cudaDataType cusolver_dtype;

template<>
cudaDataType cusolver_dtype<float> = CUDA_R_32F;

template<>
cudaDataType cusolver_dtype<double> = CUDA_R_64F;
#else
template <typename T>
rocblas_status (*rocsolver_potrf)(rocblas_handle, const rocblas_fill, const int, T*, const int, int*);

template<>
rocblas_status (*rocsolver_potrf<float>)(rocblas_handle, const rocblas_fill, const int, float*, const int, int*) = &rocsolver_spotrf;

template<>
rocblas_status (*rocsolver_potrf<double>)(rocblas_handle, const rocblas_fill, const int, double*, const int, int*) = &rocsolver_dpotrf;
#endif


template<typename T>
struct Calculator {
    cudaStream_t stream = NULL;
    cusolverDnHandle_t handle = NULL;
    int info = 0;
    int *d_info = nullptr;
#ifdef CUDA
    cusolverDnParams_t params = NULL;
    size_t d_work_size = 0;
    void *d_work = nullptr;
    size_t h_work_size = 0;
    void *h_work = nullptr;
#endif
    int n;
    int lda;
    cublasFillMode_t uplo;

    Calculator(int n, cublasFillMode_t uplo) : n{n}, lda{n}, uplo{uplo} {
        // Create handlers etc
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUSOLVER_CHECK(cusolverDnCreate(&handle));
        CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));
#ifdef CUDA
        CUSOLVER_CHECK(cusolverDnCreateParams(&params));

        // Build working space
        CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(
            handle, params, uplo, n, cusolver_dtype<T>, NULL, lda,
            cusolver_dtype<T>, &d_work_size, &h_work_size));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), d_work_size));

        if (0 < h_work_size) {
            h_work = reinterpret_cast<void *>(malloc(h_work_size));
            if (h_work == nullptr) {
                throw std::runtime_error("Error: h_work not allocated.");
            }
        }
#endif
    }

    ~Calculator() {
        // Free resources
#ifdef CUDA
        free(h_work);
        cudaFree(d_work);
#endif
        cudaFree(d_info);
        cusolverDnDestroy(handle);
        cudaStreamDestroy(stream);
    }

    void calculate(const T* d_A_input, T* A = nullptr) {
        // The input array is replaced by the output in by potrf so we work on a copy to simulate
        // what might happen in tensorflow / pytorch
        T *d_A = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(T) * n*n));
        CUDA_CHECK(cudaMemcpyAsync(d_A, d_A_input, sizeof(T) * n*n, cudaMemcpyDeviceToDevice, stream));

        // Cholesky factorization
#ifdef CUDA
        CUSOLVER_CHECK(cusolverDnXpotrf(handle, params, uplo, n, cusolver_dtype<T>,
                                        d_A, lda, cusolver_dtype<T>, d_work, d_work_size,
                                        h_work, h_work_size, d_info));
#else
        CUSOLVER_CHECK((*rocsolver_potrf<T>)(handle, uplo, n, d_A, lda, d_info));
#endif

        CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (0 > info) {
            std::stringstream ss;
            ss << -info << "-th parameter wrong";
            throw std::runtime_error(ss.str());
        }

        // Copy back to host
        if (A) {
            CUDA_CHECK(cudaMemcpyAsync(A, d_A, sizeof(T) * n*n, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        cudaFree(d_A);
    }
};


template <typename T>
void run(int n, int repeat) {
    std::cout << "RUN"
              << " n: " << n
              << " repeat: " << repeat
              << " dtype: " << typeid(T).name()
              << std::endl;

    std::vector<T> A(n*n, 0);
    std::vector<T> L(n*n, 0);

    // Build a symmetric diagonally-dominated matrix
    for (int i = 0; i < n; i++) {
        // Set off-diagonals to a value < 1
        for (int j = 0; j < n; j++) {
            A[i * n + j] = ((T)(i + j)) / (n*n);
        }
        // Set diagonal
        A[i * n + i] = i + 1;
    }

    std::cout << "Input matrix" << std::endl;
    print_matrix(n, A);

    // Copy array to device
    T *d_A = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(T) * A.size()));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeof(T) * A.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    {
        Calculator<T> calc(n, CUBLAS_FILL_MODE_LOWER); // Fast
        // Calculator<T> calc(n, CUBLAS_FILL_MODE_UPPER); // Slow

        // Warm up
        calc.calculate(d_A, L.data());

        // Zero lower half
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                L[i * n + j] = 0;
            }
        }

        std::cout << "Output matrix" << std::endl;
        print_matrix(n, L);

        // Run timing
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < repeat; iter++) {
            calc.calculate(d_A);
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> time = t1 - t0;
        std::cout << "average time " << time.count()*1e-3 / repeat << " s" << std::endl;
    }

    {
        // Run timing recreating handles etc every time
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < repeat; iter++) {
            Calculator<T> calc(n, CUBLAS_FILL_MODE_LOWER);
            calc.calculate(d_A);
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> time = t1 - t0;
        std::cout << "average time " << time.count()*1e-3 / repeat << " s (including handle creation)" << std::endl;
    }

    CUDA_CHECK(cudaFree(d_A));
}

int main(int argc, char *argv[]) {
    // Default values
    std::list<int> matrix_sizes = {10};
    int repeat = 10;
    bool do_double = true;

    // Parse args
    if (argc > 1) {
        matrix_sizes.clear();
        char *token = strtok(argv[1], ",");
        while (token != NULL) {
            matrix_sizes.push_back(std::stoi(token));
            token = strtok(NULL, ",");
        }
    }
    if (argc > 2) {
        repeat = std::stoi(argv[2]);
    }
    if (argc > 3) {
        if (std::string(argv[3]) != "double") {
            do_double = false;
        }
    }

    // Calculate
    for (auto n: matrix_sizes) {
        if (do_double)
            run<double>(n, repeat);
        else
            run<float>(n, repeat);
    }

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
