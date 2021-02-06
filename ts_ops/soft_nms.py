import torch


def soft_nms(boxes: torch.Tensor, scores: torch.Tensor, sigma: float, score_threshold: float):
    """
    Performs soft-nms on the boxes (with the (Gaussian function).

    Args:
        boxes (Tensor[N, 5]): Boxes to perform NMS on. They are expected to be in
           (x_min, y_min, x_max, y_max) format.
        scores (Tensor[N]): Scores for each one of the boxes
        sigma (float): The sigma parameter described in the paper which controls 
        score_threshold: 
    Returns:
        updated_scores (Tensor): float tensor with the updated scores, i.e.
            the scores after they have been decreased according to the overlap, 
            sorted in decreasing order of scores
        keep (Tensor): int64 tensor with the indices of the elements that have been kept
            by soft-nms, sorted in decreasing order of scores
    """

    return torch.ops.ts_ops.soft_nms(boxes, scores, sigma, score_threshold)
