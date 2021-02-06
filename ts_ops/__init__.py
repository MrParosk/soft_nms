from .soft_nms import soft_nms
import os
import torch

this_dir = os.path.dirname(__file__)

# TODO: see why the .so files is installed below our package folder
torch.ops.load_library(os.path.join(this_dir, "..", "soft_nms.so"))
