from typing import Tuple

import torch


@torch.library.register_fake("soft_nms::soft_nms")  # type: ignore
def soft_nms_abstract(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    sigma: float,
    score_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (torch.empty_like(scores), torch.empty_like(scores, dtype=torch.int32))
