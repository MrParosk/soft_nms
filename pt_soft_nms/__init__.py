from .soft_nms import soft_nms, batched_soft_nms
import os
import torch

# TODO: see why the .so files is installed below our package folder
if os.name == "nt":
    file = "soft_nms.pyd"
else:
    file = "soft_nms.so"

this_dir = os.path.dirname(__file__)
torch.ops.load_library(os.path.join(this_dir, "..", file))
