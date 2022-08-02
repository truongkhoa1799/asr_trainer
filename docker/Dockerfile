FROM nvcr.io/nvidia/pytorch:21.12-py3

USER root
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
    build-essential libboost-all-dev \
    cmake libsndfile1 sox \
    zlib1g-dev libbz2-dev liblzma-dev \
    libfreetype6 python-setuptools swig \
    python-dev ffmpeg git wget apt-utils \
    libasound-dev portaudio19-dev python3-pyaudio \
    && rm -rf /var/lib/apt/lists/*

# uninstall stuff from base container
RUN pip uninstall -y sacrebleu torchtext

WORKDIR /home/khoatlv/pckgs
COPY NeMo /home/khoatlv/pckgs/NeMo
RUN cd NeMo && pip3 install -e .

COPY decoders /home/khoatlv/pckgs/decoders
RUN cd decoders && bash setup_kenlm.sh
RUN cd decoders && bash setup_openfst.sh
RUN cd decoders && bash setup_threadpool.sh
RUN cd decoders && python3 setup.py install --num_processes 4

RUN pip3 uninstall -y setuptools transformers huggingface-hub numba numpy
RUN pip3 install llvmlite==0.38.0 --ignore-installed
RUN pip3 install setuptools==59.5.0 transformers==4.9.2 huggingface-hub==0.0.12 numpy==1.19.5 numba==0.55.1
RUN pip3 install https://github.com/kpu/kenlm/archive/master.zip
RUN pip3 uninstall pesq -y && cd ..

COPY python-pesq /home/khoatlv/pckgs/python-pesq
RUN cd python-pesq && pip3 install .
RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

WORKDIR /home/khoatlv/pckgs
COPY portaudio /home/khoatlv/pckgs/portaudio
RUN cd portaudio && chmod 777 configure && ./configure && make && make install

WORKDIR /home/khoatlv/pckgs
COPY requirements.txt /home/khoatlv/pckgs/requirements.txt
RUN pip3 install --disable-pip-version-check --no-cache-dir -r requirements.txt

WORKDIR /home/khoatlv