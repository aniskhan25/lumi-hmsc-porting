// kernel_example.cu.cc
#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "kernel_example.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "magma_v2.h"
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <stdexcept>
#include <iostream>
#include <cmath>
using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

// Define the CUDA kernel.
template <typename T>
__global__ void ExampleCudaKernel(const int size, const T* in, T* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    out[i] = 2 * __ldg(in + i);
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
void ExampleFunctor<GPUDevice, T>::operator()(
    const GPUDevice& d, int size, const T* in, T* out) {
  magma_init();
  
  // Init custream, might be good to test magmaalloc
  cudaStream_t stream = NULL;
  int n = sqrt(size); // Just for testing
  int info = 0;

  magma_int_t d_info;

  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  // Allocate array on GPU
  double *d_A = nullptr;

  // Copy over input array
  cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * size);
  cudaMemcpy(d_A, in, sizeof(double) * size, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  // I would look into this way of doing things. Started below, but haven't tested
  //magma_int_t ldda = magma_roundup( n, 32 );  // round up to multiple of 32 for best GPU performance
  //magma_malloc(size, sizeof(double));
  //magma_dsetmatrix(n,n,size,sizeof(double), 


  magma_dpotrf_expert_gpu(MagmaLower, n, d_A, n, &d_info, 1024, MagmaNative);

  // Copy info
  cudaMemcpyAsync(&info, &d_info, sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  if (0 != info) {
     std::stringstream ss;
     ss << -info << "-th parameter wrong";
     throw std::runtime_error(ss.str());
  }

  // Copy to out array
  cudaMemcpyAsync(out, d_A, sizeof(double) * n*n, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
}

// Explicitly instantiate functors for the types of OpKernels registered.
// template struct ExampleFunctor<GPUDevice, float>;
template struct ExampleFunctor<GPUDevice, double>;

#endif  // GOOGLE_CUDA

