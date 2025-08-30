# C++ Development Environment Setup

## Initial Setup Commands

Run these commands to get your development environment ready:

```bash
# Make scripts executable
chmod +x dev.sh scripts/pre-commit.py

# Setup development environment (creates directories, installs git hooks)
./dev.sh setup

# Build the development container
./dev.sh build

# Start the container
./dev.sh start

# Open development shell
./dev.sh shell
```

## Inside the Container

Once you're in the container shell, you can:

```bash
# Configure and build the project
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build

# Run code quality checks
clang-format -i $(find src include -name "*.cpp" -o -name "*.hpp" -o -name "*.h")
clang-tidy $(find src include -name "*.cpp" -o -name "*.c") -p build

# Run tests
ctest --test-dir build --output-on-failure
```

## Available Tools in Container

- CMake 3.20+
- Ninja build system  
- Clang 18 (compiler, tidy, format)
- Cppcheck
- Valgrind
- GDB/LLDB debuggers
- Code coverage tools (lcov, gcovr)
- Python 3 with development packages

## Project Structure Created by Setup

```
T07-ADS_Lane-Detection/
├── src/                    # Your source files (.cpp)
├── include/                # Your header files (.hpp, .h)
├── tests/                  # Unit test files
├── build/                  # CMake build directory
├── scripts/                # Development scripts
├── .github/workflows/      # CI/CD workflows
├── CMakeLists.txt          # Build configuration
├── .clang-format          # Code style configuration
├── .clang-tidy            # Static analysis rules
└── dev.sh                 # Development helper script
```

The setup script will create a sample `src/main.cpp` to get you started.
