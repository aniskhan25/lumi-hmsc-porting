#ifndef MAGMA_CHOLESKY_H_
#define MAGMA_CHOLESKY_H_

#include <unsupported/Eigen/CXX11/Tensor>

template <typename Device, typename T>
struct MagmaCholeskyFunctor {
  void operator()(const Device& d, int size, const T* in, T* out);
};

#if TENSORFLOW_USE_ROCM
template <typename T>
struct MagmaCholeskyFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d, int size, const T* in, T* out);
 };
#endif

#endif MAGMA_CHOLESKY_H_
