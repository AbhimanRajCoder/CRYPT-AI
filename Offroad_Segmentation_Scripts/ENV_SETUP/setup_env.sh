#!/bin/bash

# Call the first shell script
echo "Running create_env.sh..."
chmod +x create_env.sh
./create_env.sh

# Call the second shell script after the first completes
echo "Running install_packages.sh..."
chmod +x install_packages.sh
./install_packages.sh

echo "All tasks completed."
