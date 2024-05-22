#ifdef TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#include "magma_cholesky.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "magma_v2.h"
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <iostream>
#include <cmath>
using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

template <typename T>
void MagmaCholeskyFunctor<GPUDevice, T>::operator()(
    const GPUDevice& d, int size, const T* in, T* out) {
  magma_init();
  
  int n = sqrt(size); // Just for testing
  int info = 0;
  hipStream_t stream = NULL;
  magma_int_t d_info;

  hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
  hipMalloc(reinterpret_cast<void **>(&d_info), sizeof(int));


  // Copy over input to out and call potrf there.
  hipMemcpy(out, in, sizeof(double) * size, hipMemcpyDeviceToDevice);
  hipDeviceSynchronize();

  // I would look into this way of doing things. Started below, but haven't tested
  //magma_int_t ldda = magma_roundup( n, 32 );  // round up to multiple of 32 for best GPU performance
  //magma_malloc(size, sizeof(double));
  //magma_dsetmatrix(n,n,size,sizeof(double), 


  magma_dpotrf_expert_gpu(MagmaLower, n, out, n, &d_info, 1024, MagmaNative);

  // Copy info
  hipMemcpyAsync(&info, &d_info, sizeof(int), hipMemcpyDeviceToHost, stream);
  hipStreamSynchronize(stream);

  if (0 != info) {
     std::stringstream ss;
     ss << -info << "-th parameter wrong";
     throw std::runtime_error(ss.str());
  }

  hipFree(&d_info);
  hipStreamDestroy(stream);
};

// Explicitly instantiate functors for the types of OpKernels registered.
// template struct MagmaCholeskyFunctor<GPUDevice, float>;
template struct MagmaCholeskyFunctor<GPUDevice, double>;

#endif  // GOOGLE_CUDA

