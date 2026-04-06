#!/bin/bash

# Check if python3 is available in the system
if ! command -v python3 &> /dev/null
then
    echo "Python 3 is not found in your system. Please install it first."
    exit 1
fi

# Create the virtual environment folder 'EDU_env'
echo "Creating the Python virtual environment 'EDU_env'..."
python3 -m venv EDU_env

if [ $? -eq 0 ]; then
    echo "Virtual environment created successfully in $(pwd)/EDU_env"
else
    echo "Failed to create virtual environment."
    exit 1
fi
