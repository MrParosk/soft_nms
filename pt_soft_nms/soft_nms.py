import torch


def soft_nms(boxes: torch.Tensor, scores: torch.Tensor, sigma: float, score_threshold: float = 0.0):
    """
    Performs soft-nms on the boxes (with the Gaussian function).

    Args:
        boxes (Tensor[N, 5]): Boxes to perform NMS on. They are expected to be in
           (x_min, y_min, x_max, y_max) format.
        scores (Tensor[N]): Scores for each one of the boxes
        sigma (float): The sigma parameter described in the paper which controls
        score_threshold: Will filter out all updated-scores which has value than score_threshold
    Returns:
        updated_scores (Tensor): float tensor with the updated scores, i.e.
            the scores after they have been decreased according to the overlap,
            sorted in decreasing order of scores
        keep (Tensor): int64 tensor with the indices of the elements that have been kept
            by soft-nms, sorted in decreasing order of scores
    """

    assert len(
        boxes.shape) == 2 and boxes.shape[-1] == 4, f"boxes has wrong shape, expected (N, 4), got {boxes.shape}"
    assert len(
        scores.shape) == 1, f"scores has wrong shape, expected (N,) got {scores.shape}"

    return torch.ops.soft_nms.soft_nms(boxes, scores, sigma, score_threshold)
