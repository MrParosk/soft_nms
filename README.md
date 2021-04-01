# Soft-nms in PyTorch

![master](https://github.com/MrParosk/soft_nms/workflows/master/badge.svg?branch=master) [![codecov](https://codecov.io/gh/MrParosk/soft_nms/branch/master/graph/badge.svg?token=VWTV2Q54XR)](https://codecov.io/gh/MrParosk/soft_nms)

Implementation of the soft-nms algorithm described in the paper: [Soft-NMS -- Improving Object Detection With One Line of Code](https://arxiv.org/abs/1704.04503)

The algorithm is implemented in PyTorch's C++ frontend for better performance.

## Install

Make sure that you have installed PyTorch, version 1.7 or higher. Install the package by

```Shell
pip install git+https://github.com/MrParosk/soft_nms.git
```

Note that if you are using Windows, you need MSVC installed.
