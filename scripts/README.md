# Cholesky benchmarking

## C++ code

See [cholesky.cpp](cholesky.cpp).

### LUMI (MI250X / 1 GCD)

#### Available rocm versions before Sep 2024 upgrade

```bash
ml rocm/5.2.3

hipcc -std=c++14 --offload-arch=gfx90a -O3 -lrocblas -lrocsolver cholesky.cpp

srun -p dev-g --nodes=1 --ntasks-per-node=1 --mem=32G --gpus-per-node=1 -t 00:15:00 ./a.out 3,25000 10

RUN n: 3 repeat: 10 dtype: d
Input matrix
    1.000    0.111    0.222
    0.111    2.000    0.333
    0.222    0.333    3.000
Output matrix
    1.000    0.000    0.000
    0.111    1.410    0.000
    0.222    0.219    1.704
average time 7.7099e-05 s
average time 0.019707 s (including handle creation)
RUN n: 25000 repeat: 10 dtype: d
average time 9.91697 s
average time 9.9373 s (including handle creation)
```

```bash
ml LUMI/22.08
ml partition/G
ml rocm/5.3.3

hipcc -std=c++14 --offload-arch=gfx90a -O3 -lrocblas -lrocsolver cholesky.cpp

srun -p dev-g --nodes=1 --ntasks-per-node=1 --mem=32G --gpus-per-node=1 -t 00:15:00 ./a.out 3,25000 10

RUN n: 3 repeat: 10 dtype: d
Input matrix
    1.000    0.111    0.222
    0.111    2.000    0.333
    0.222    0.333    3.000
Output matrix
    1.000    0.000    0.000
    0.111    1.410    0.000
    0.222    0.219    1.704
average time 5.46878e-05 s
average time 0.0202523 s (including handle creation)
RUN n: 25000 repeat: 10 dtype: d
average time 1.22292 s
average time 1.24275 s (including handle creation)
```

#### Available rocm versions after Sep 2024 upgrade

```bash
ml rocm/6.0.3

hipcc -std=c++14 --offload-arch=gfx90a -O3 -lrocblas -lrocsolver cholesky.cpp

srun -p dev-g --nodes=1 --ntasks-per-node=1 --mem=32G --gpus-per-node=1 -t 00:15:00 ./a.out 3,25000 10

RUN n: 3 repeat: 10 dtype: d
Input matrix
    1.000    0.111    0.222
    0.111    2.000    0.333
    0.222    0.333    3.000
Output matrix
    1.000    0.000    0.000
    0.111    1.410    0.000
    0.222    0.219    1.704
average time 7.26577e-05 s
average time 0.000550958 s (including handle creation)
RUN n: 25000 repeat: 10 dtype: d
average time 1.24172 s
average time 1.24306 s (including handle creation)
```


#### Singularity rocm versions before Sep 2024 upgrade

Installing:
```bash
export EBU_USER_PREFIX=$WORKDIR/EasyBuild
ml LUMI
ml partition/container
ml EasyBuild-user

eb rocm-5.6.1-singularity-20240207.eb
mkdir $EBU_USER_PREFIX/SW/container/rocm/5.6.1-singularity-20240207/runscripts

eb rocm-5.7.1-singularity-20240207.eb
mkdir $EBU_USER_PREFIX/SW/container/rocm/5.7.1-singularity-20240207/runscripts
```


```bash
export EBU_USER_PREFIX=$WORKDIR/EasyBuild
ml LUMI/22.08
ml partition/G
ml rocm/5.6.1-singularity-20240207

singularity exec $SIF hipcc -std=c++14 --offload-arch=gfx90a -O3 -lrocblas -lrocsolver cholesky.cpp

srun -p dev-g --nodes=1 --ntasks-per-node=1 --mem=32G --gpus-per-node=1 -t 00:15:00 singularity exec $SIF ./a.out 3,25000 10

RUN n: 3 repeat: 10 dtype: d
Input matrix
    1.000    0.111    0.222
    0.111    2.000    0.333
    0.222    0.333    3.000
Output matrix
    1.000    0.000    0.000
    0.111    1.410    0.000
    0.222    0.219    1.704
average time 5.6816e-05 s
average time 0.000574949 s (including handle creation)
RUN n: 25000 repeat: 10 dtype: d
average time 1.22186 s
average time 1.22325 s (including handle creation)
```

```bash
export EBU_USER_PREFIX=$WORKDIR/EasyBuild
ml LUMI/22.08
ml partition/G
ml rocm/5.7.1-singularity-20240207

singularity exec $SIF hipcc -std=c++14 --offload-arch=gfx90a -O3 -lrocblas -lrocsolver cholesky.cpp

srun -p dev-g --nodes=1 --ntasks-per-node=1 --mem=32G --gpus-per-node=1 -t 00:15:00 singularity exec $SIF ./a.out 3,25000 10

RUN n: 3 repeat: 10 dtype: d
Input matrix
    1.000    0.111    0.222
    0.111    2.000    0.333
    0.222    0.333    3.000
Output matrix
    1.000    0.000    0.000
    0.111    1.410    0.000
    0.222    0.219    1.704
average time 6.91339e-05 s
average time 0.000627651 s (including handle creation)
RUN n: 25000 repeat: 10 dtype: d
average time 1.2161 s
average time 1.21686 s (including handle creation)
```

#### Singularity rocm versions after Sep 2024 upgrade

```bash
export SINGULARITY_CACHEDIR=$PWD/singularity_cache
singularity pull docker://docker.io/rocm/dev-ubuntu-22.04:6.2-complete
export SINGULARITY_BIND="/pfs,/scratch,/projappl,/project,/flash,/appl"

singularity exec dev-ubuntu-20.04_6.2-complete.sif hipcc -std=c++14 --offload-arch=gfx90a -O3 -lrocblas -lrocsolver cholesky.cpp

srun -p dev-g --nodes=1 --ntasks-per-node=1 --mem=32G --gpus-per-node=1 -t 00:15:00 singularity exec dev-ubuntu-22.04_6.2-complete.sif ./a.out 3,25000 10

srun: job 8033257 queued and waiting for resources
srun: job 8033257 has been allocated resources
RUN n: 3 repeat: 10 dtype: d
Input matrix
   1.000000e+00   1.111111e-01   2.222222e-01
   1.111111e-01   2.000000e+00   3.333333e-01
   2.222222e-01   3.333333e-01   3.000000e+00
Output matrix
   1.000000e+00   0.000000e+00   0.000000e+00
   1.111111e-01   1.409842e+00   0.000000e+00
   2.222222e-01   2.189196e-01   1.703729e+00
average time 4.2706e-05 s
average time 0.000545893 s (including handle creation)
RUN n: 25000 repeat: 10 dtype: d
Input matrix
   1.000000e+00   1.600000e-09   3.200000e-09   4.800000e-09            ...   3.999360e-05   3.999520e-05   3.999680e-05   3.999840e-05
   1.600000e-09   2.000000e+00   4.800000e-09   6.400000e-09            ...   3.999520e-05   3.999680e-05   3.999840e-05   4.000000e-05
   3.200000e-09   4.800000e-09   3.000000e+00   8.000000e-09            ...   3.999680e-05   3.999840e-05   4.000000e-05   4.000160e-05
   4.800000e-09   6.400000e-09   8.000000e-09   4.000000e+00            ...   3.999840e-05   4.000000e-05   4.000160e-05   4.000320e-05
            ...            ...            ...            ...            ...            ...            ...            ...            ...
   3.999360e-05   3.999520e-05   3.999680e-05   3.999840e-05            ...   2.499700e+04   7.998880e-05   7.999040e-05   7.999200e-05
   3.999520e-05   3.999680e-05   3.999840e-05   4.000000e-05            ...   7.998880e-05   2.499800e+04   7.999200e-05   7.999360e-05
   3.999680e-05   3.999840e-05   4.000000e-05   4.000160e-05            ...   7.999040e-05   7.999200e-05   2.499900e+04   7.999520e-05
   3.999840e-05   4.000000e-05   4.000160e-05   4.000320e-05            ...   7.999200e-05   7.999360e-05   7.999520e-05   2.500000e+04
Output matrix
   1.000000e+00   0.000000e+00   0.000000e+00   0.000000e+00            ...   0.000000e+00   0.000000e+00   0.000000e+00   0.000000e+00
   1.600000e-09   1.414214e+00   0.000000e+00   0.000000e+00            ...   0.000000e+00   0.000000e+00   0.000000e+00   0.000000e+00
   3.200000e-09   3.394113e-09   1.732051e+00   0.000000e+00            ...   0.000000e+00   0.000000e+00   0.000000e+00   0.000000e+00
   4.800000e-09   4.525483e-09   4.618802e-09   2.000000e+00            ...   0.000000e+00   0.000000e+00   0.000000e+00   0.000000e+00
            ...            ...            ...            ...            ...            ...            ...            ...            ...
   3.999360e-05   2.828088e-05   2.309216e-05   1.999920e-05            ...   1.581044e+02   0.000000e+00   0.000000e+00   0.000000e+00
   3.999520e-05   2.828201e-05   2.309309e-05   2.000000e-05            ...   5.057904e-07   1.581076e+02   0.000000e+00   0.000000e+00
   3.999680e-05   2.828314e-05   2.309401e-05   2.000080e-05            ...   5.058005e-07   5.058005e-07   1.581107e+02   0.000000e+00
   3.999840e-05   2.828427e-05   2.309493e-05   2.000160e-05            ...   5.058106e-07   5.058106e-07   5.058106e-07   1.581139e+02
average time 0.757435 s
average time 0.758504 s (including handle creation)
```


### Puhti (V100)

```bash
ml cuda/11.7.0

nvcc -DCUDA -O3 -arch=sm_70 -lcusolver cholesky.cpp

srun -p gputest --nodes=1 --ntasks-per-node=1 --mem=32G --gres=gpu:v100:1 -t 0:15:00 ./a.out 3,25000 10

RUN n: 3 repeat: 10 dtype: d
Input matrix
    1.000    0.111    0.222
    0.111    2.000    0.333
    0.222    0.333    3.000
Output matrix
    1.000    0.000    0.000
    0.111    1.410    0.000
    0.222    0.219    1.704
average time 5.00553e-05 s
average time 0.000919892 s (including handle creation)
RUN n: 25000 repeat: 10 dtype: d
average time 0.844338 s
average time 0.848023 s (including handle creation)
```


### Mahti (A100)

```bash
ml cuda/11.5.0

nvcc -DCUDA -O3 -arch=sm_80 -lcusolver cholesky.cpp

srun -p gputest --nodes=1 --ntasks-per-node=1 --mem=32G --gres=gpu:a100:1 -t 0:15:00 ./a.out 3,25000 10

RUN n: 3 repeat: 10 dtype: d
Input matrix
    1.000    0.111    0.222
    0.111    2.000    0.333
    0.222    0.333    3.000
Output matrix
    1.000    0.000    0.000
    0.111    1.410    0.000
    0.222    0.219    1.704
average time 4.43281e-05 s
average time 0.00102775 s (including handle creation)
RUN n: 25000 repeat: 10 dtype: d
average time 0.395884 s
average time 0.397762 s (including handle creation)
```


### LUMI MAGMA

```bash
COMPILE:
export SINGULARITY_BIND="/pfs,/scratch,/projappl,/project,/flash,/appl"
singularity exec /appl/local/containers/sif-images/lumi-pytorch-rocm-5.5.1-python-3.10-pytorch-v2.0.1.sif hipcc -std=c++14 --offload-arch=gfx90a -O3 -I/opt/magma/include/ -lrocblas -lrocsolver -L/opt/magma/lib -Wl,-rpath,/opt/magma/lib -lmagma -o magma magma.cpp

RUN:
srun -p dev-g --nodes=1 --ntasks-per-node=1 --mem=32G --gpus-per-node=1 -t 00:15:00 singularity exec /appl/local/containers/sif-images/lumi-pytorch-rocm-5.5.1-python-3.10-pytorch-v2.0.1.sif ./magma 3,25000 10

RUN n: 3 repeat: 10 dtype: d
Input matrix
    1.000    0.111    0.222
    0.111    2.000    0.333
    0.222    0.333    3.000
Output matrix
    1.000    0.000    0.000
    0.111    1.410    0.000
    0.222    0.219    1.704
average time 0.00119727 s
average time 0.00167095 s (including handle creation)
RUN n: 25000 repeat: 10 dtype: d
average time 0.377096 s
average time 0.378239 s (including handle creation)
```


### Mahti MAGMA

```bash
export SINGULARITY_BIND="/scratch,/projappl,/appl"
singularity exec -B /local_scratch ~/magma.sif nvcc -DCUDA -O3 -arch=sm_80 -I/opt/magma/include/ -L/opt/magma/lib -lmagma -Xcompiler \"-Wl,-rpath,/opt/magma/lib\" -o magma magma.cpp

srun -p gputest --nodes=1 --ntasks-per-node=1 --mem=32G --gres=gpu:a100:1 -t 0:15:00 singularity exec --nv ~/magma.sif ./magma 3,25000 10

```

Compiling magma natively:
```bash
ml cuda/11.5.0

git clone --branch v2.8.0 https://bitbucket.org/icl/magma
cd magma
grep -rl '^#!/usr/bin/env python$' . | xargs sed -i 's|^#!/usr/bin/env python$|#!/usr/bin/env python3|g'
cp make.inc-examples/make.inc.openblas make.inc

# Launch interactive shell on a node
srun -p test --nodes=1 --ntasks-per-node=1 --cpus-per-task=128 --exclusive -t 1:00:00 --pty bash

export TMPDIR=/dev/shm
make -j128 lib/libmagma.so GPU_TARGET=Ampere OPENBLASDIR=$OPENBLAS_INSTALL_ROOT CUDADIR=$CUDA_INSTALL_ROOT
```


### A100 MAGMA
Using CUDA/12.1.0 and gcc/12.1.0, installed with conda MKL backend according to Samuel's instructions.

```bash
RUN n: 3 repeat: 10 dtype: d
Input matrix
    1.000    0.111    0.222
    0.111    2.000    0.333
    0.222    0.333    3.000
Output matrix
    1.000    0.000    0.000
    0.111    1.410    0.000
    0.222    0.219    1.704
average time 5.81113e-05 s
average time 0.00155553 s (including handle creation)
RUN n: 25000 repeat: 10 dtype: d
average time 0.324008 s
average time 0.327149 s (including handle creation)
```

