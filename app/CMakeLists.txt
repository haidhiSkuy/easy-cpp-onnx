cmake_minimum_required(VERSION 3.12)
project(onnxruntime_cpp)




set(ONNXRUNTIME_ROOTDIR "/usr/local")
include_directories("${ONNXRUNTIME_ROOTDIR}/include"                           # Pre-built package
                    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime"               # Linux local install to /usr/local
                    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session")

link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(onnxruntime_cpp main.cpp)

# Link against ONNX Runtime
target_link_libraries(onnxruntime_cpp onnxruntime) 
target_link_libraries(onnxruntime_cpp ${OpenCV_LIBS})