import torch
# torch.ops.load_library("build/libts_ops.so")
#from ops_cpp import soft_nms

print(torch.ops)
func = torch.ops.ts_ops.soft_nms
# print(func)


def soft_nms(dets: torch.Tensor, scores: torch.Tensor, sigma: float, iou_threshold: float):
    """
    TBC.
    """

    return torch.ops.ts_ops.soft_nms(dets, scores, sigma, iou_threshold)
