import torch


def soft_nms(dets: torch.Tensor, scores: torch.Tensor, sigma: float, iou_threshold: float):
    """
    TBC.
    """

    return torch.ops.ts_ops.soft_nms(dets, scores, sigma, iou_threshold)
