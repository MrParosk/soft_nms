from typing import Tuple

import torch
import torch.lib


@torch.library.register_fake("soft_nms::soft_nms")  # type: ignore
def soft_nms_abstract(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    sigma: float,
    score_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ctx = torch.library.get_ctx()
    nnz = ctx.new_dynamic_size()
    return (scores.new_empty(nnz), scores.new_empty(nnz, dtype=torch.int64))
