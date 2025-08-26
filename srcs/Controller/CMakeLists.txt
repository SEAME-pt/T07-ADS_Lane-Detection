cmake_minimum_required(VERSION 3.10)
project(Controller)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)


find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GPIOD REQUIRED libgpiod)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -Wextra")

# Includes do projeto e sysroot
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/includes
    ${CMAKE_CURRENT_SOURCE_DIR}/eigen
    /workspace/libraries/CppAD                # <- Inclui o CppAD copiado
    /opt/jetson-sysroot/usr/include
    /opt/jetson-sysroot/usr/include/aarch64-linux-gnu
    /opt/jetson-sysroot/usr/local/cuda/include
    /opt/jetson-sysroot/usr/local/include/opencv4
    /opt/jetson-sysroot/usr/local/include/opencv4/opencv2
    /opt/jetson-sysroot/usr/src/tensorrt/include
    /opt/jetson-sysroot/usr/lib/aarch64-linux-gnu/tegra
    /opt/jetson-sysroot/usr/include/SDL2
    /opt/jetson-sysroot/usr/local/include
)

# Bibliotecas externas
find_library(I2C_LIB            i2c              PATHS /opt/jetson-sysroot/usr/lib/aarch64-linux-gnu)
find_library(NVINFER            nvinfer          PATHS /opt/jetson-sysroot/usr/lib/aarch64-linux-gnu)
find_library(NVONNXPARSER       nvonnxparser     PATHS /opt/jetson-sysroot/usr/lib/aarch64-linux-gnu)
find_library(CUDART             cudart           PATHS /opt/jetson-sysroot/usr/local/cuda/lib64)
find_library(ZMQ                zmq              PATHS /opt/jetson-sysroot/usr/lib/aarch64-linux-gnu)
find_library(SDL2_LIB           SDL2             PATHS /opt/jetson-sysroot/usr/lib/aarch64-linux-gnu)
find_library(NVMEDIA            nvmedia          PATHS /opt/jetson-sysroot/usr/lib/aarch64-linux-gnu/tegra)
find_library(IPOPT_LIB          ipopt            PATHS /opt/jetson-sysroot/usr/lib/aarch64-linux-gnu /opt/jetson-sysroot/usr/local/lib)
# find_library(LAPACK_LIB         lapack           PATHS /opt/jetson-sysroot/usr/lib/aarch64-linux-gnu)
# find_library(BLAS_LIB           blas             PATHS /opt/jetson-sysroot/usr/lib/aarch64-linux-gnu)
# find_library(LAPACK_LIB         lapack           NAMES liblapack.so.3.7.1 lapack PATHS /opt/jetson-sysroot/usr/lib/aarch64-linux-gnu/lapack)
# find_library(BLAS_LIB           blas             NAMES libblas.so.3.7.1 blas PATHS /opt/jetson-sysroot/usr/lib/aarch64-linux-gnu/blas)

set(LAPACK_LIB "/opt/jetson-sysroot/usr/lib/aarch64-linux-gnu/lapack/liblapack.so.3.7.1")
set(BLAS_LIB "/opt/jetson-sysroot/usr/lib/aarch64-linux-gnu/blas/libblas.so.3.7.1")

# find_library(LAPACK_LIB lapack PATHS /opt/jetson-sysroot/usr/lib/aarch64-linux-gnu/lapack NO_DEFAULT_PATH)
# find_library(BLAS_LIB   blas   PATHS /opt/jetson-sysroot/usr/lib/aarch64-linux-gnu/blas NO_DEFAULT_PATH)
find_library(M_LIB m PATHS /opt/jetson-sysroot/usr/lib/aarch64-linux-gnu NO_DEFAULT_PATH)
# Checagens
if(NOT IPOPT_LIB)
    message(FATAL_ERROR "IPOPT library not found")
endif()
if(NOT LAPACK_LIB)
    message(FATAL_ERROR "LAPACK library not found")
endif()
if(NOT BLAS_LIB)
    message(FATAL_ERROR "BLAS library not found")
endif()

# OpenCV
find_library(OpenCV_CORE        opencv_core        PATHS /opt/jetson-sysroot/usr/local/lib)
find_library(OpenCV_IMGPROC     opencv_imgproc     PATHS /opt/jetson-sysroot/usr/local/lib)
find_library(OpenCV_HIGHGUI     opencv_highgui     PATHS /opt/jetson-sysroot/usr/local/lib)
find_library(OpenCV_VIDEOIO     opencv_videoio     PATHS /opt/jetson-sysroot/usr/local/lib)
find_library(OpenCV_VIDEO       opencv_video       PATHS /opt/jetson-sysroot/usr/local/lib)
find_library(OpenCV_CUDAWARPING opencv_cudawarping PATHS /opt/jetson-sysroot/usr/local/lib)
find_library(OpenCV_IMGCODECS   opencv_imgcodecs   PATHS /opt/jetson-sysroot/usr/local/lib)

# Fontes do projeto + temp_file.cpp do CppAD
set(SOURCES
    /workspace/libraries/CppAD/cppad_lib/temp_file.cpp   # <- Essencial para evitar "undefined reference"
    sources/Controller.cpp
    sources/SpeedPIDController.cpp
    sources/JetCar.cpp
    sources/devices.cpp
    sources/LaneDetector.cpp
    sources/MPC.cpp
    sources/Debug.cpp
    sources/main.cpp
)

message(STATUS "LAPACK_LIB: ${LAPACK_LIB}")


# Executável
add_executable(ControllerExec ${SOURCES})

# Linkagem
target_link_libraries(ControllerExec
    Threads::Threads
    ${GPIOD_LIBRARIES}
    ${I2C_LIB}
    ${NVINFER}
    ${NVONNXPARSER}
    ${CUDART}
    ${ZMQ}
    ${SDL2_LIB}
    ${NVMEDIA}
    ${IPOPT_LIB}
    ${LAPACK_LIB}
    ${BLAS_LIB}
    ${M_LIB}
    ${OpenCV_CUDAWARPING}
    ${OpenCV_VIDEOIO}
    ${OpenCV_VIDEO}
    ${OpenCV_IMGCODECS}
    ${OpenCV_HIGHGUI}
    ${OpenCV_IMGPROC}
    ${OpenCV_CORE}
)

