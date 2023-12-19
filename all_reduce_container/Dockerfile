FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt update && apt install -y libibverbs1 libibverbs-dev ibutils libibmad-dev libibumad-dev rdma-core librdmacm1 librdmacm-dev librdmacm-dev librdmacm1 libibumad-dev perftest libmlx5-1 && apt-get install -y wget vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && /bin/bash ~/miniconda.sh -b -p /opt/conda

ENV PATH=/opt/conda/bin:$PATH

RUN conda create -n all-reduce-bench -y -c nvidia/label/cuda-11.8.1 -c conda-forge -c pytorch -c defaults magma-cuda121 astunparse numpy ninja pyyaml setuptools cmake typing_extensions six requests dataclasses mkl mkl-include pip python=3.11.1

RUN pip install torch torchvision

WORKDIR /
#RUN conda activate all-reduce-bench
SHELL ["conda", "run", "-n", "all-reduce-bench", "/bin/bash", "-c"]
COPY all_reduce_bench.py /