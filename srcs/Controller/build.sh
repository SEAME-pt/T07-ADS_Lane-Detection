#!/bin/bash
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=../toolchain-aarch64.cmake -DCMAKE_BUILD_TYPE=Release
if [ $? -ne 0 ]; then
	echo "CMake configuration failed."
	exit 1
fi
make -j$(nproc)
if [ $? -ne 0 ]; then
	echo "Make failed."
	exit 1
fi
echo "Build completed successfully."
# Copy the built binary to the parent directory
scp ControllerExec team07@10.21.221.47:/home/team07/car_services
