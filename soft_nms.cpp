#include <torch/torch.h>
// #include <torch/slicing.h>
#include <iostream>

torch::Tensor soft_nms(
    const torch::Tensor &dets,
    const torch::Tensor &scores,
    const float iou_threshold)
{

    int num_element = dets.sizes()[0];
    torch::Tensor indicies = torch::arange(0, num_element, torch::dtype(torch::kFloat32)).view({num_element, 1});
    auto dets_indicies = torch::cat({dets, indicies}, 1);

    auto x1 = dets_indicies.index({torch::indexing::Slice(), 0});
    auto y1 = dets_indicies.index({torch::indexing::Slice(), 1});
    auto x2 = dets_indicies.index({torch::indexing::Slice(), 2});
    auto y2 = dets_indicies.index({torch::indexing::Slice(), 3});
    auto areas = (x2 - x1 + 1) * (y2 - y1 + 1);

    for (int i = 0; i < num_element; i++)
    {
    }

    return dets_indicies;
};

int main()
{
    auto dets = torch::tensor({{400, 200, 200, 200}, {400, 200, 200, 200}});
    auto scores = torch::tensor({0.9, 0.5});
    auto x = soft_nms(dets, scores, 0.5);
}
