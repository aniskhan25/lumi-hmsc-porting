# Cholesky benchmarking

## C++ code

See [cholesky.cpp](cholesky.cpp).

### LUMI (MI250X / 1 GCD)

#### Available rocm versions

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

#### Singularity rocm versions

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

