#!/bin/bash

set -e

# ATM cmake is only used for running cpp-tests

current_dir=$(pwd)
mkdir -p build
(cd build && cmake -DCMAKE_PREFIX_PATH="$current_dir/libtorch" -DCURRENT_DIR=$current_dir .. && make -j)

./build/tests/cpp/run_tests
