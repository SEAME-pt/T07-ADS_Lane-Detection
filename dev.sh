#!/bin/bash

# Development container management script

CONTAINER_NAME="cpp-dev-container"
COMPOSE_FILE="docker-compose.dev.yml"

show_help() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build     Build the development container"
    echo "  start     Start the development container"
    echo "  shell     Open a shell in the running container"
    echo "  stop      Stop the development container"
    echo "  restart   Restart the development container"
    echo "  clean     Remove container and rebuild"
    echo "  format    Run clang-format on all source files"
    echo "  tidy      Run clang-tidy on all source files"
    echo "  setup     Setup git hooks and initial configuration"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build     # Build the container"
    echo "  $0 start     # Start container in background"
    echo "  $0 shell     # Open interactive shell"
    echo "  $0 format    # Format all code"
}

build_container() {
    echo "Building development container..."
    docker-compose -f $COMPOSE_FILE build
}

start_container() {
    echo "Starting development container..."
    docker-compose -f $COMPOSE_FILE up -d
    echo "Container started. Use '$0 shell' to enter."
}

open_shell() {
    if ! docker-compose -f $COMPOSE_FILE ps | grep -q "Up"; then
        echo "Container not running. Starting..."
        start_container
        sleep 2
    fi
    echo "Opening shell in development container..."
    docker-compose -f $COMPOSE_FILE exec cpp-dev bash
}

stop_container() {
    echo "Stopping development container..."
    docker-compose -f $COMPOSE_FILE down
}

restart_container() {
    echo "Restarting development container..."
    stop_container
    start_container
}

clean_container() {
    echo "Cleaning development container..."
    docker-compose -f $COMPOSE_FILE down
    docker-compose -f $COMPOSE_FILE build --no-cache
}

run_clang_format() {
    echo "Running clang-format on all source files..."
    if ! docker-compose -f $COMPOSE_FILE ps | grep -q "Up"; then
        echo "Starting container for formatting..."
        start_container
        sleep 2
    fi
    
    docker-compose -f $COMPOSE_FILE exec cpp-dev bash -c "
        find . -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.c' | \
        grep -E '\.(src|include)/' | \
        xargs clang-format -i
    "
    echo "✓ clang-format completed"
}

run_clang_tidy() {
    echo "Running clang-tidy on all source files..."
    if ! docker-compose -f $COMPOSE_FILE ps | grep -q "Up"; then
        echo "Starting container for linting..."
        start_container
        sleep 2
    fi
    
    docker-compose -f $COMPOSE_FILE exec cpp-dev bash -c "
        if [ ! -f build/compile_commands.json ]; then
            echo 'Building project first...'
            cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
            cmake --build build
        fi
        
        find src include -name '*.cpp' -o -name '*.c' 2>/dev/null | \
        xargs clang-tidy -p build
    "
}

setup_hooks() {
    echo "Setting up git hooks and development environment..."
    
    # Install pre-commit hook
    if [ -f .git/hooks/pre-commit ]; then
        echo "Backing up existing pre-commit hook..."
        mv .git/hooks/pre-commit .git/hooks/pre-commit.backup
    fi
    
    cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook for C++ code quality

# Check if we're in a container or have tools available locally
if command -v clang-format &> /dev/null && command -v clang-tidy &> /dev/null; then
    # Run locally
    python3 scripts/pre-commit.py
else
    # Run in container
    echo "Running pre-commit checks in container..."
    docker-compose -f docker-compose.dev.yml exec cpp-dev python3 scripts/pre-commit.py
fi
EOF
    
    chmod +x .git/hooks/pre-commit
    echo "✓ Pre-commit hook installed"
    
    # Create initial project structure if it doesn't exist
    mkdir -p src include tests
    
    if [ ! -f src/main.cpp ]; then
        cat > src/main.cpp << 'EOF'
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
EOF
        echo "✓ Created sample src/main.cpp"
    fi
    
    echo "✓ Setup completed"
}

# Main command handling
case "$1" in
    build)
        build_container
        ;;
    start)
        start_container
        ;;
    shell)
        open_shell
        ;;
    stop)
        stop_container
        ;;
    restart)
        restart_container
        ;;
    clean)
        clean_container
        ;;
    format)
        run_clang_format
        ;;
    tidy)
        run_clang_tidy
        ;;
    setup)
        setup_hooks
        ;;
    help|--help|-h)
        show_help
        ;;
    "")
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use '$0 help' for available commands."
        exit 1
        ;;
esac
