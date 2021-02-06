#include "soft_nms.h"

torch::Tensor calculate_area(const torch::Tensor &boxes)
{
    auto x1 = boxes.index({torch::indexing::Slice(), 0});
    auto y1 = boxes.index({torch::indexing::Slice(), 1});
    auto x2 = boxes.index({torch::indexing::Slice(), 2});
    auto y2 = boxes.index({torch::indexing::Slice(), 3});
    auto areas = (x2 - x1) * (y2 - y1);
    return areas;
}

torch::Tensor calculate_iou(const torch::Tensor &boxes, const torch::Tensor &areas, const int i)
{
    auto xx1 = torch::maximum(boxes.index({i, 0}),
                              boxes.index({torch::indexing::Slice(i + 1, torch::indexing::None), 0}));

    auto yy1 = torch::maximum(boxes.index({i, 1}),
                              boxes.index({torch::indexing::Slice(i + 1, torch::indexing::None), 1}));

    auto xx2 = torch::minimum(boxes.index({i, 2}),
                              boxes.index({torch::indexing::Slice(i + 1, torch::indexing::None), 2}));

    auto yy2 = torch::minimum(boxes.index({i, 3}),
                              boxes.index({torch::indexing::Slice(i + 1, torch::indexing::None), 3}));

    auto w = torch::maximum(torch::zeros_like(xx1), xx2 - xx1);
    auto h = torch::maximum(torch::zeros_like(yy1), yy2 - yy1);

    auto intersection = w * h;
    auto union_ = areas.index({i}) + areas.index({torch::indexing::Slice(i + 1, torch::indexing::None)}) - intersection;
    auto iou = torch::div(intersection, union_);
    return iou;
}

void update_sorting_order(torch::Tensor &boxes, torch::Tensor &scores, torch::Tensor &areas, const int idx)
{
    // Since the scores get updated with soft-max we need to "re-sort"

    torch::Tensor max_score, t_max_idx;
    std::tie(max_score, t_max_idx) = torch::max(scores.index({torch::indexing::Slice(idx + 1, torch::indexing::None)}), 0);

    // max_idx is computed from sliced data, therefore need to convert it to "global" max idx
    auto max_idx = t_max_idx.item<int>() + idx + 1;

    if (scores.index({idx}).item<float>() < max_score.item<float>())
    {
        auto boxes_idx = boxes.index({idx}).clone();
        auto boxes_max = boxes.index({max_idx}).clone();
        boxes.index({idx}) = boxes_max;
        boxes.index({max_idx}) = boxes_idx;

        auto scores_idx = scores.index({idx}).clone();
        auto scores_max = scores.index({max_idx}).clone();
        scores.index({idx}) = scores_max;
        scores.index({max_idx}) = scores_idx;

        auto areas_idx = areas.index({idx}).clone();
        auto areas_max = areas.index({max_idx}).clone();
        areas.index({idx}) = areas_max;
        areas.index({max_idx}) = areas_idx;
    }
}

std::tuple<torch::Tensor, torch::Tensor> soft_nms(
    const torch::Tensor &boxes,
    const torch::Tensor &scores,
    const double sigma,
    const double score_threshold)
{
    int num_element = boxes.sizes()[0];
    auto indicies = torch::arange(0, num_element, torch::dtype(torch::kFloat32)).view({num_element, 1});
    indicies = indicies.to(boxes.device());
    auto boxes_indicies = torch::cat({boxes, indicies}, 1);

    auto scores_updated = scores.clone();
    auto areas = calculate_area(boxes_indicies);

    for (int i = 0; i < num_element - 1; i++)
    {
        update_sorting_order(boxes_indicies, scores_updated, areas, i);
        auto iou = calculate_iou(boxes_indicies, areas, i);
        auto weight = torch::exp(-(iou * iou) / sigma);
        scores_updated.index({torch::indexing::Slice(i + 1, torch::indexing::None)}) = weight * scores_updated.index({torch::indexing::Slice(i + 1, torch::indexing::None)});
    }

    auto keep = boxes_indicies.index({scores_updated > score_threshold, 4});
    keep = keep.to(torch::kLong);

    scores_updated = scores_updated.index({scores_updated > score_threshold});
    return std::make_tuple(scores_updated, keep);
}
