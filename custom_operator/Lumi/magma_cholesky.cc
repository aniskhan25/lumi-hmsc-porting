#include "magma_cholesky.h"
#include <iostream>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <Python.h>

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("MagmaCholesky")
    .Attr("T: numbertype")
    .Input("arr: T")
    .Input("n: int32")
    .Output("factorized: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status();
    });

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class MagmaCholeskyOp : public OpKernel {
 public:
  explicit MagmaCholeskyOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    const Tensor& n_tensor = context->input(1);
    int n = n_tensor.flat<int>()(0);

    
    int input_dims = input_tensor.dims();

    OP_REQUIRES(context, input_dims == 3 || input_dims == 2 ,
                errors::InvalidArgument("Input tensor must be 2 or 3-dimensional"));

    int num_matrices = 1;
    int ldda = input_tensor.dim_size(1);
    if (input_dims==3){
    	num_matrices = input_tensor.dim_size(0);
	ldda = input_tensor.dim_size(2);
    }
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    std::cout << "LDDA " << ldda << " N " << n << "\n";
    MagmaCholeskyFunctor<Device, T>()(
        context->eigen_device<Device>(),
	n,
	ldda,
        input_tensor.flat<T>().data(),
        output_tensor->flat<T>().data(),
        num_matrices);
  }
};

// Register the GPU kernels.
#ifdef TENSORFLOW_USE_ROCM
#define REGISTER_GPU(T)                                          \
  extern template class MagmaCholeskyFunctor<GPUDevice, T>;            \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("MagmaCholesky").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      MagmaCholeskyOp<GPUDevice, T>);
REGISTER_GPU(double);

#endif
// Python initialization function
extern "C" PyMODINIT_FUNC PyInit_magma_cholesky(void) {
    static PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "magma_cholesky",
        nullptr,
        -1,
        nullptr,
    };

    PyObject* module = PyModule_Create(&module_def);
    if (module == nullptr) {
        return nullptr;
    }

    return module;
}

