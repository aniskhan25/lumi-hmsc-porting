# TensorFlow custom operator for MAGMA cholesky

The custom operator here implements cholesky factorization using MAGMA for single matrices and arrays of matrices.

## Custom Operators in TensorFlow

Information about TensorFlow custom operators can be found almost exclusively here:
https://www.tensorflow.org/guide/create_op

As a summary, there are 3 components for creating a GPU operation in TensorFlow:
1. Definition of the operation, available data types and method of execution (CPU/GPU) - magma_cholesky.cc
2. Header files for the CPU/GPU operation - magma_cholesky.h
3. The GPU kernels - magma_cholesky.cu.cc

For the operation, the input arrays already reside on the GPU and are consts. This means they are first transferred to
output variable, on which the in place factorization is performed. Another option of explicitly keeping the input on
host memory was tried, but it was significantly slower.

## Compiling the operator into a usable library

For compilation it is necessary to have MAGMA libraries and TensorFlow available. Access to GPUs is not necessary during compilation.
On LUMI, MAGMA is already made available in a container. The only necessary addition to this is having a compatible version of TensorFlow.
As of August 2024, MAGMA is present in the container located at this path `/appl/local/containers/sif-images/lumi-pytorch-rocm-5.5.1-python-3.10-pytorch-v2.0.1.sif`.
As it now stands, all containers provided by LUMI will be published in the directory `/appl/local/containers/sif-images/`.
The names give an overview of the versions included in the container, in this case being ROCm5.5.1, Python3.10 and PyTorch2.0.1.
MAGMA is installed in the container at `/opt`, which will likely also be the case in the future. ROCm or Python changes should not break the custom operator, but in case there are updates to the syntax of TensorFlow or MAGMA, all is possible.

### Installing ROCm compatible TensorFlow in the container:

```bash
#Enter the container in interactive mode
singularity shell /appl/local/containers/sif-images/lumi-pytorch-rocm-5.5.1-python-3.10-pytorch-v2.0.1.sif

#Activate the conda environment inside the container
$WITH_CONDA

#Install the TensorFlow version compatible with ROCm5.5.1. https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/tensorflow-install.html
pip install tensorflow-rocm==2.11.1.550 
```

### Compilation
The compilation of the operation into a dynamic library can be done in the following way. Most of these steps are described in the TensorFlow documentation linked above. The first 3 steps are necessary only once and are the most time consuming. The whole process should take up to a few minutes.
```bash
# Entering the container and setting base flags as before
singularity shell /appl/local/containers/sif-images/lumi-pytorch-rocm-5.5.1-python-3.10-pytorch-v2.0.1.sif #Enter the container in interactive mode
$WITH_CONDA #Activate the conda environment inside the container
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/magma/lib #Make MAGMA libraries available at runtime

# Getting TensorFlow libraries and headers for compiling and linking the TensorFlow components in the Op.
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))'))
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

# Compiling the GPU kernels into the object file magma_cholesky.cu.o
hipcc -std=c++14 -c -o magma_cholesky.cu.o magma_cholesky.cu.cc   ${TF_CFLAGS[@]} -D EIGEN_USE_HIP=1 -D TENSORFLOW_USE_ROCM=1 -fPIC -I/opt/magma/include -L/opt/magma/lib -I/opt
hipcc -std=c++14 -c -o magma_cholesky.cu.o magma_cholesky.cu.cc   ${TF_CFLAGS[@]} -D EIGEN_USE_HIP=1 -D TENSORFLOW_USE_ROCM=1 -fPIC -I/scratch/project_462000008/tiks/MAGMA/include -L/scratch/project_462000008/tiks/MAGMA/lib6.2 -I/opt

# Compiling the final library magma_cholesky.so
gcc-10 -std=c++14 -shared -o magma_cholesky.so magma_cholesky.cc   magma_cholesky.cu.o ${TF_CFLAGS[@]} -fPIC  ${TF_LFLAGS[@]} -I/opt/magma/include -L/opt/magma/lib -lmagma -D TENSORFLOW_USE_ROCM=1
gcc-10 -std=c++14 -shared -o magma_cholesky.so magma_cholesky.cc   magma_cholesky.cu.o ${TF_CFLAGS[@]} -fPIC  ${TF_LFLAGS[@]} -I/scratch/project_462000008/tiks/MAGMA/include -L/scratch/project_462000008/tiks/MAGMA/lib6.2 -lmagma -D TENSORFLOW_USE_ROCM=1
```
## Using the Operation in Python

To actually use the final library `magma_cholesky.so` inside of a Python script we have to import it. It look like this
```bash
# We load the library via a relative of absolute path and save it under magma_lib in this case.
magma_lib=tf.load_op_library('./magma_cholesky.so')

# We define our operation's name in the operation definition - magma_cholesky.cc - file. The op is registered there on line 13, but gets automatically converted to snakecase.
cholesky = magma_lib.magma_cholesky

# Now that we have the function saved as cholesky, we can simply use it by providing an input to it.
result = cholesky(input_matrix)
```

The previous steps are applied in the code `custom_operator_cholesky.py`. Given a compiled library in the same folder, the
code can be executed interactively like so, starting from the login node.
```bash
# Get an interactive shell
srun -psmall-g --gpus 1 --pty bash

singularity shell /appl/local/containers/sif-images/lumi-pytorch-rocm-5.5.1-python-3.10-pytorch-v2.0.1.sif
$WITH_CONDA 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/magma/lib

# The command below will give input to the operator as an array of 10 matrices
python3 custom_operator_cholesky.py 10,100 10 double true


# The command below will give input to the operator as 10 separate matrices
python3 custom_operator_cholesky.py 10,100 10 double false
```

## Using the operation in HMSC 

To use the library in HMSC, we currently manually replace specific cholesky function calls with the Magma implementation in the HMSC source. The calls we replace are:

```
https://github.com/aniskhan25/hmsc-hpc/blob/3dfb9310f4cc238a41031e232c8e9389ddfc16e6/hmsc/updaters/updateBetaLambda.py#L120
https://github.com/aniskhan25/hmsc-hpc/blob/3dfb9310f4cc238a41031e232c8e9389ddfc16e6/hmsc/updaters/updateEta.py#L118
https://github.com/aniskhan25/hmsc-hpc/blob/3dfb9310f4cc238a41031e232c8e9389ddfc16e6/hmsc/updaters/updateEta.py#L133
https://github.com/aniskhan25/hmsc-hpc/blob/3dfb9310f4cc238a41031e232c8e9389ddfc16e6/hmsc/updaters/updateEta.py#L137
https://github.com/aniskhan25/hmsc-hpc/blob/3dfb9310f4cc238a41031e232c8e9389ddfc16e6/hmsc/updaters/updateBetaLambda.py#L94
```

To run HMSC using the modified Cholesky, first follow the above instructions to build `magma_cholesky.so`.

Then, in each file referenced above, load the custom `.so` by adding the following code:

```
magma_lib=tf.load_op_library('/path/to/magma_cholesky.so')
```

And then modify the referenced function calls above, for example:

```
LiUEta = magma_lib.magma_cholesky(iUEta)
```

Once the modifications are complete, launch the container as above while in the HMSC base directory (as `$PWD` will be mounted automatically - if you launch from some other directory you will need to bind mount the HMSC directory so you can access the code).

Install the pip requirements for HMSC

TODO: The below requirements shoudl work, but need to double check installed versions of all these packages once LUMI upgrade is complete
```
tensorflow-probability==0.19.0
MySQL_python==1.2.2 [TODO: double check - is this one necessary?]
numpy==1.22.4
scipy==1.13
ujson
pandas
pyreadr
```
Exit the container, then you should be able to run HSMC via interactive shell as above:

```bash
# Get an interactive shell
srun -psmall-g --gpus 1 --pty bash

singularity shell /appl/local/containers/sif-images/lumi-pytorch-rocm-5.5.1-python-3.10-pytorch-v2.0.1.sif
$WITH_CONDA 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/magma/lib

N_SAMPLES=...
N_TRANS=...
N_THIN=...

python3 -m hmsc.run_gibbs_sampler --input examples/big_spatial/init_1fu_ns622_ny01600_chain01.rds --output $PWD/output.rds --samples N_SAMPLES --transient N_TRANS --thin N_THIN --verbose 100
```
