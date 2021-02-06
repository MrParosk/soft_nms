#pragma once
#include <torch/torch.h>
#include <tuple>

torch::Tensor calculate_area(const torch::Tensor &dets);

torch::Tensor calculate_iou(const torch::Tensor &dets, const torch::Tensor &areas, const int i);

std::tuple<torch::Tensor, torch::Tensor> create_sorted_dets(const torch::Tensor &dets, const torch::Tensor &scores);

torch::Tensor soft_nms(
    const torch::Tensor &dets,
    const torch::Tensor &scores,
    const double sigma,
    const double iou_threshold);
