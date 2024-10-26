FROM ubuntu:20.04 

# compiler, cmake, git
RUN apt update -y \ 
&& apt upgrade -y && \
apt install g++ -y && \ 
apt install cmake -y && \ 
apt install git -y

# Onnxruntime
copy onnxruntime-linux-x64-gpu-1.19.2.tgz . 
RUN tar -zxvf onnxruntime-linux-x64-gpu-1.19.2.tgz -C /opt
RUN cp -r /opt/onnxruntime-linux-x64-gpu-1.19.2/include /usr/local && \
    cp -r /opt/onnxruntime-linux-x64-gpu-1.19.2/lib /usr/local 

# Opencv 
RUN git clone https://github.com/opencv/opencv.git /opt/opencv
RUN git clone https://github.com/opencv/opencv_contrib.git /opt/opencv_contrib
RUN cd /opt/opencv && \ 
    mkdir build && \ 
    cd build && \ 
    cmake -D WITH_CUDA=ON \
          -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
          -D CUDA_ARCH_BIN=8.6 \
          -D CUDA_ARCH_PTX=8.6 \
          -D WITH_CUBLAS=1 \
           .. && \     
    make -j$(nproc) && \ 
    make install