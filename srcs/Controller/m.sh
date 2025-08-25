#!/bin/sh

# Run the make command
make

# Check the exit status of make
if [ $? -ne 0 ]; then
    echo "Makefile error detected. Stopping execution."
    exit 1
fi

# If make succeeds, you can add more commands here if needed
scp ControllerExec team07@100.96.63.44:/home/team07/car_services
ls -l ControllerExec
