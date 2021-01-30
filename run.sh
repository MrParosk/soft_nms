libtorch_path=/home/erik/git/ts_ops/libtorch/

(cd build && cmake -DCMAKE_PREFIX_PATH=$libtorch_path .. && cmake --build . --config Release && ./ts_ops)
