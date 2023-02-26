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

## Example usage

```python
import torch

from pt_soft_nms import batched_soft_nms, soft_nms

sigma = 0.5
score_threshold = 0.1

boxes = torch.tensor([[20, 20, 40, 40], [10, 10, 20, 20], [20, 20, 35, 35]], device="cpu", dtype=torch.float)
scores = torch.tensor([0.5, 0.9, 0.11], device="cpu", dtype=torch.float)
updated_scores, keep = soft_nms(boxes, scores, sigma, score_threshold)
# updated_scores=tensor([0.9000, 0.5000]), keep=tensor([1, 0])

# With batched_soft_nms, the soft-nms will be applied per batch, which is specified with indicies
indicies = torch.tensor([0, 0, 1], device="cpu")
keep_batch = batched_soft_nms(boxes, scores, indicies, sigma, score_threshold)
# keep_batch=tensor([1, 0, 2])
```
