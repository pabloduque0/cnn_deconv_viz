FROM nvidia/cuda

RUN apt-get -y update && apt-get -y install \
        python3 \
        python3-dev \
        git \
        wget \
        unzip \
        cmake \
        build-essential \
        pkg-config \
        libatlas-base-dev \
        gfortran \
        libgtk2.0-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libv4l-dev \
        python3-pip \
        python3-setuptools \
        fsl

RUN pip3 install --upgrade pip

RUN  pip3 install --upgrade setuptools==41.0.0

RUN python3 -m pip install numpy SimpleITK scipy sklearn opencv-python matplotlib \
                            itk tensorflow-gpu keras tensorflow_addons

ADD src/ wmhseg_example/