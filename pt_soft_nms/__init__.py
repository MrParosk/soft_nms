import os

import torch

from .soft_nms import batched_soft_nms, soft_nms  # noqa: F401

if os.name == "nt":
    file = "soft_nms.pyd"
else:
    file = "soft_nms.so"

# TODO: see why the .so files is installed below our package folder
this_dir = os.path.dirname(__file__)
torch.ops.load_library(os.path.join(this_dir, "..", file))  # type: ignore
