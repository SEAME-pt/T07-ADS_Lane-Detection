cmake_minimum_required(VERSION 3.10)
project(Controller)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)

# Find required packages
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GPIOD REQUIRED libgpiod)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -Wextra")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/includes
    ${CMAKE_CURRENT_SOURCE_DIR}/eigen
    /usr/include
    /usr/include/aarch64-linux-gnu
    /usr/local/cuda/include
    /usr/local/include/opencv4
    /usr/local/include/opencv4/opencv2
    /usr/src/tensorrt/include
    /usr/lib/aarch64-linux-gnu/tegra
    /usr/include/SDL2
    /usr/local/include
)

# Find libraries
find_library(I2C_LIB            i2c              PATHS /usr/lib/aarch64-linux-gnu)
find_library(NVINFER            nvinfer          PATHS /usr/lib/aarch64-linux-gnu)
find_library(NVONNXPARSER       nvonnxparser     PATHS /usr/lib/aarch64-linux-gnu)
find_library(CUDART             cudart           PATHS /usr/local/cuda/lib64)
find_library(ZMQ                zmq              PATHS /usr/lib/aarch64-linux-gnu)
find_library(SDL2_LIB           SDL2             PATHS /usr/lib/aarch64-linux-gnu)
find_library(NVMEDIA            nvmedia          PATHS /usr/lib/aarch64-linux-gnu/tegra)
find_library(IPOPT_LIB          ipopt            PATHS /usr/lib/aarch64-linux-gnu /usr/local/lib)
find_library(CPPAD_LIB          cppad_lib        PATHS /usr/lib/aarch64-linux-gnu /usr/local/lib)
find_library(LAPACK_LIB         lapack           PATHS /usr/lib/aarch64-linux-gnu)
find_library(BLAS_LIB           blas             PATHS /usr/lib/aarch64-linux-gnu)

# OpenCV modules
find_library(OpenCV_CORE        opencv_core        PATHS /usr/local/lib)
find_library(OpenCV_IMGPROC     opencv_imgproc     PATHS /usr/local/lib)
find_library(OpenCV_HIGHGUI     opencv_highgui     PATHS /usr/local/lib)
find_library(OpenCV_VIDEOIO     opencv_videoio     PATHS /usr/local/lib)
find_library(OpenCV_VIDEO       opencv_video       PATHS /usr/local/lib)
find_library(OpenCV_CUDAWARPING opencv_cudawarping PATHS /usr/local/lib)
find_library(OpenCV_IMGCODECS   opencv_imgcodecs   PATHS /usr/local/lib)

# Check if libraries were found
if(NOT IPOPT_LIB)
    message(FATAL_ERROR "IPOPT library not found")
endif()
if(NOT CPPAD_LIB)
    message(FATAL_ERROR "CppAD library not found")
endif()
if(NOT LAPACK_LIB)
    message(FATAL_ERROR "LAPACK library not found")
endif()
if(NOT BLAS_LIB)
    message(FATAL_ERROR "BLAS library not found")
endif()

# Source files
set(SOURCES
    sources/Controller.cpp
    sources/SpeedPIDController.cpp
    sources/JetCar.cpp
    sources/devices.cpp
    sources/LaneDetector.cpp
    sources/MPC.cpp
    sources/Debug.cpp
    sources/main.cpp
)

# Executable
add_executable(ControllerExec ${SOURCES})

# Link libraries
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
    ${CPPAD_LIB}
    ${LAPACK_LIB}
    ${BLAS_LIB}
    ${OpenCV_CUDAWARPING}
    ${OpenCV_VIDEOIO}
    ${OpenCV_VIDEO}
    ${OpenCV_IMGCODECS}
    ${OpenCV_HIGHGUI}
    ${OpenCV_IMGPROC}
    ${OpenCV_CORE}
)