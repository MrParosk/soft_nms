from typing import List, Tuple

import torch


def soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    sigma: float,
    score_threshold: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs soft-nms on the boxes (with the Gaussian function).

    Args:
        boxes (Tensor[N, 4]): Boxes to perform NMS on. They are expected to be in
           (x_min, y_min, x_max, y_max) format.
        scores (Tensor[N]): Scores for each one of the boxes.
        sigma (float): The sigma parameter described in the paper which controls how much the score is
            decreased on overlap.
        score_threshold (float): Will filter out all updated-scores which has value than score_threshold.
    Returns:
        updated_scores (Tensor): float tensor with the updated scores, i.e.
            the scores after they have been decreased according to the overlap,
            sorted in decreasing order of scores
        keep (Tensor): int64 tensor with the indices of the elements that have been kept
            by soft-nms, sorted in decreasing order of scores
    """

    assert len(boxes.shape) == 2 and boxes.shape[-1] == 4, f"boxes has wrong shape, expected (N, 4), got {boxes.shape}"
    assert len(scores.shape) == 1, f"scores has wrong shape, expected (N,) got {scores.shape}"

    return torch.ops.soft_nms.soft_nms(boxes, scores, sigma, score_threshold)  # type: ignore


def batched_soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    idxs: torch.Tensor,
    sigma: float,
    score_threshold: float = 0.0,
) -> torch.Tensor:
    """
    Performs soft non-maximum suppression in a batched fashion.
    Each index value correspond to a category, and soft-nms
    will not be applied between elements of different categories.
    Args:
        boxes (Tensor[N, 4]):
           boxes where soft-nms will be performed. They
           are expected to be in (x_min, y_min, x_max, y_max) format.
        scores (Tensor[N]):
           scores for each one of the boxes.
        idxs (Tensor[N]):
           indices of the categories for each one of the boxes.
        sigma (float): The sigma parameter described in the paper which controls how much the score is
            decreased on overlap.
        score_threshold (float): Will filter out all updated-scores which has value than score_threshold.
    Returns:
        keep (Tensor): int64 tensor with the indices of the elements that have been kept
            by soft-nms, sorted in decreasing order of scores
    """

    assert len(boxes.shape) == 2 and boxes.shape[-1] == 4, f"boxes has wrong shape, expected (N, 4), got {boxes.shape}"
    assert len(scores.shape) == 1, f"scores has wrong shape, expected (N,) got {scores.shape}"

    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for id in torch.jit.annotate(List[int], torch.unique(idxs).cpu().tolist()):  # type: ignore
        mask = torch.nonzero(idxs == id).view(-1)
        _, keep = soft_nms(boxes[mask], scores[mask], sigma, score_threshold)
        result_mask[mask[keep]] = True
    keep = torch.nonzero(result_mask).view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep
