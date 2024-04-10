ARG CONDA_VERSION=py310_24.1.2-0

FROM docker.io/nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qy && \
    apt-get install -qy \
        curl \
        && \
    apt-get clean

ARG CONDA_VERSION
RUN curl https://repo.anaconda.com/miniconda/Miniconda3-$CONDA_VERSION-Linux-x86_64.sh -o conda.sh && \
    bash conda.sh -b -p /conda && \
    rm conda.sh && \
    /conda/bin/conda clean -afy

RUN . /conda/etc/profile.d/conda.sh && \
    conda create -p /conda/env \
        python=3.10 \
        && \
    /conda/bin/conda clean -afy

RUN . /conda/etc/profile.d/conda.sh && \
    conda activate /conda/env && \
    conda install --only-deps pytorch && \
    conda install ninja pillow mkl-include && \
    /conda/bin/conda clean -afy

RUN apt-get update -qy && \
    apt-get install -qy \
        build-essential \
        gfortran \
        git \
        && \
    apt-get clean

ENV PATH=/conda/env/bin:$PATH \
    LC_ALL=C.UTF-8

# based on https://github.com/sfantao/lumi-containers/blob/main/pytorch/build-rocm-5.5.1-python-3.10-pytorch-v2.0.1.docker
RUN mkdir -p /opt/magma && \
    git clone --depth 1 --branch v2.8.0 https://bitbucket.org/icl/magma /opt/magma-build && \
    cd /opt/magma-build && \
    \
    cp make.inc-examples/make.inc.mkl-gcc make.inc && \
    echo 'LIBDIR += -L$(MKLROOT)/lib' >> make.inc && \
    echo 'LIB += -Wl,--enable-new-dtags -Wl,--rpath,\$(MKLROOT)/lib -Wl,--rpath,/opt/magma/lib' >> make.inc ; \
    nice make lib/libmagma.so GPU_TARGET=Ampere MKLROOT=/conda/env CUDADIR=/usr/local/cuda -j8 && \
    cp -rf include lib /opt/magma && \
    cd / && rm -rf /opt/magma-build

ENTRYPOINT ["bash"]
