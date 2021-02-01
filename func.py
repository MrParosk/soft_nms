import unittest
import torch
torch.ops.load_library("build/libts_ops.so")

dets = torch.tensor(
    [[20, 20, 40, 40], [10, 10, 20, 20], [20, 20, 35, 35]]).float()
scores = torch.tensor([0.5, 0.9, 0.11]).float()


def my_func(dets, scores):
    return torch.ops.ts_ops.soft_nms(dets, scores, 0.5, 0.05)


_ = torch.jit.script(my_func, dets, scores)
