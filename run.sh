libtorch_path="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')"

(cd build && cmake -DCMAKE_PREFIX_PATH=$libtorch_path .. && make -j)
