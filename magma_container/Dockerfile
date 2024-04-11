FROM docker.io/nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qy && \
    apt-get install -qy \
        build-essential \
        gfortran \
        libopenblas-openmp-dev \
        python3 \
        python-is-python3 \
        git \
        && \
    apt-get clean

# based on https://github.com/sfantao/lumi-containers/blob/main/pytorch/build-rocm-5.5.1-python-3.10-pytorch-v2.0.1.docker
RUN mkdir -p /opt/magma && \
    git clone --depth 1 --branch v2.8.0 https://bitbucket.org/icl/magma /opt/magma-build && \
    cd /opt/magma-build && \
    cp make.inc-examples/make.inc.openblas make.inc && \
    nice make lib/libmagma.so GPU_TARGET=Ampere CUDADIR=/usr/local/cuda OPENBLASDIR=/usr/lib/x86_64-linux-gnu/openblas-openmp -j4 && \
    cp -rf include lib /opt/magma && \
    cd / && rm -rf /opt/magma-build

ENV LC_ALL=C.UTF-8

ENTRYPOINT ["bash"]
