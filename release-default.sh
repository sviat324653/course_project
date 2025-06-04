#!/bin/bash

# Exit on error
set -e

# Define build directory
BUILD_DIR=build-release
APPLICATION_BIN=full

# Create the build directory if it doesn't exist
if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating build directory..."
    mkdir "$BUILD_DIR"
fi

#gnome-terminal -- bash -c " \
#	gnome-terminal  --tab -- bash -c 'cd $BUILD_DIR && cmake .. ; exec bash;' && \
#	gnome-terminal  --tab -- bash -c 'cd $BUILD_DIR && cmake --build . ; exec bash;' && \
#	gnome-terminal  --tab -- bash -c 'cd $BUILD_DIR && ./$APPLICATION_BIN; exec bash;'"
	
	
#gnome-terminal --window -- bash -c "cd $BUILD_DIR && cmake .. && cmake --build . && gnome-terminal --tab -- bash -c './$APPLICATION_BIN; exec bash;'; exec bash;"

gnome-terminal --window -- bash -c "cd $BUILD_DIR && cmake -G Ninja -DCMAKE_BUILD_TYPE=Release .. && gnome-terminal --tab -- bash -c \"cmake --build . && gnome-terminal --tab -- bash -c \\\"./$APPLICATION_BIN; exit_code=\\\\\\\$?; echo; echo; echo; echo -e \\\\\\\"\033[1;31mProcess finished with exit code \\\\\\\$exit_code\033[0m\\\\\\\"; echo; exec bash;\\\"; exec bash;\"; exec bash;"

	





