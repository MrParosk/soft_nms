import torch
torch.ops.load_library("build/libts_ops.so")


def soft_nms(dets: torch.Tensor, scores: torch.Tensor, sigma: float, iou_threshold: float):
    """
    TBC.
    """

    return torch.ops.ts_ops.soft_nms(dets, scores, sigma, iou_threshold)
