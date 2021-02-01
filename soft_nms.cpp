#include <torch/torch.h>
#include <tuple>

torch::Tensor calculate_area(const torch::Tensor &dets)
{
    auto x1 = dets.index({torch::indexing::Slice(), 0});
    auto y1 = dets.index({torch::indexing::Slice(), 1});
    auto x2 = dets.index({torch::indexing::Slice(), 2});
    auto y2 = dets.index({torch::indexing::Slice(), 3});
    auto areas = (x2 - x1) * (y2 - y1);
    return areas;
}

torch::Tensor calculate_iou(const torch::Tensor &dets, const torch::Tensor &areas, const int i)
{
    auto xx1 = torch::maximum(dets.index({i, 0}),
                              dets.index({torch::indexing::Slice(i + 1, torch::indexing::None), 0}));

    auto yy1 = torch::maximum(dets.index({i, 1}),
                              dets.index({torch::indexing::Slice(i + 1, torch::indexing::None), 1}));

    auto xx2 = torch::minimum(dets.index({i, 2}),
                              dets.index({torch::indexing::Slice(i + 1, torch::indexing::None), 2}));

    auto yy2 = torch::minimum(dets.index({i, 3}),
                              dets.index({torch::indexing::Slice(i + 1, torch::indexing::None), 3}));

    auto w = torch::maximum(torch::zeros_like(xx1), xx2 - xx1);
    auto h = torch::maximum(torch::zeros_like(yy1), yy2 - yy1);

    auto intersection = w * h;
    auto union_ = areas.index({i}) + areas.index({torch::indexing::Slice(i + 1, torch::indexing::None)}) - intersection;
    auto iou = torch::div(intersection, union_);
    return iou;
}

std::tuple<torch::Tensor, torch::Tensor> create_sorted_dets(const torch::Tensor &dets, const torch::Tensor &scores)
{
    int num_element = dets.sizes()[0];

    // Need the indicies since we will sort the tensors. However want to return the original indicies.
    auto indicies = torch::arange(0, num_element, torch::dtype(torch::kFloat32)).view({num_element, 1});
    auto dets_indicies = torch::cat({dets, indicies}, 1);

    // The soft-nms assumes that dets are sorted by scores
    torch::Tensor sorted_scores, sort_indicies;
    std::tie(sorted_scores, sort_indicies) = torch::sort(scores, 0, true);
    auto sorted_dets = dets_indicies.index({sort_indicies});
    return std::tie(sorted_dets, sorted_scores);
}

torch::Tensor soft_nms(
    const torch::Tensor &dets,
    const torch::Tensor &scores,
    const double sigma,
    const double iou_threshold)
{
    torch::Tensor sorted_dets, sorted_scores;
    std::tie(sorted_dets, sorted_scores) = create_sorted_dets(dets, scores);
    auto areas = calculate_area(sorted_dets);

    int num_element = dets.sizes()[0];
    for (int i = 0; i < num_element; i++)
    {
        auto iou = calculate_iou(sorted_dets, areas, i);
        auto weight = torch::exp(-(iou * iou) / sigma);
        sorted_scores.index({torch::indexing::Slice(i + 1, torch::indexing::None)}) = weight * sorted_scores.index({torch::indexing::Slice(i + 1, torch::indexing::None)});
    }

    auto keep = sorted_dets.index({torch::indexing::Slice(), 4}).index({sorted_scores > iou_threshold});
    keep = keep.to(torch::kInt);
    return keep;
}

TORCH_LIBRARY(ts_ops, m)
{
    m.def("soft_nms", soft_nms);
}
