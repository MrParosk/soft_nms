import torch


def soft_nms(boxes: torch.Tensor, scores: torch.Tensor, sigma: float, iou_threshold: float):
    """
    TBC.
    """

    return torch.ops.ts_ops.soft_nms(boxes, scores, sigma, iou_threshold)
