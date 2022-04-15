#pragma once
#include <tuple>
#include <torch/torch.h>


torch::Tensor calculate_area(const torch::Tensor &boxes);


torch::Tensor calculate_iou(const torch::Tensor &boxes, const torch::Tensor &areas, const int i);


void update_sorting_order(torch::Tensor &boxes, torch::Tensor &scores, torch::Tensor &areas, const int idx);


std::tuple<torch::Tensor, torch::Tensor> soft_nms(
    const torch::Tensor &boxes,
    const torch::Tensor &scores,
    const double sigma,
    const double score_threshold);
