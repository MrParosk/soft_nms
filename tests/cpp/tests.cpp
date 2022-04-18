#include <gtest/gtest.h>
#include "../../pt_soft_nms/csrc/soft_nms.h"

TEST(calculate_area, single_box) {
    auto boxes = torch::tensor({1, 2, 3, 4}, {torch::kFloat32});
    boxes = boxes.view({1, 4});
    auto areas = calculate_area(boxes);

    auto expected_areas = torch::tensor({4}, {torch::kFloat32});
    ASSERT_TRUE(torch::equal(expected_areas, areas));
}

TEST(calculate_area, multiple_boxes) {
    auto boxes = torch::tensor({1, 2, 3, 4, 5, 6, 8, 9}, {torch::kFloat32});
    boxes = boxes.view({2, 4});
    torch::Tensor areas = calculate_area(boxes);

    auto expected_areas = torch::tensor({4, 9}, {torch::kFloat32});
    ASSERT_TRUE(torch::equal(expected_areas, areas));
}

TEST(calculate_iou, iou_overlap) {
    auto boxes = torch::tensor({1, 2, 3, 4, 1, 2, 2, 3}, {torch::kFloat32});
    boxes = boxes.view({2, 4});
    auto areas = torch::tensor({4, 1}, {torch::kFloat32});
    int idx = 0;
    torch::Tensor ious = calculate_iou(boxes, areas, idx);

    auto expected_ious = torch::tensor({0.25}, {torch::kFloat32});
    ASSERT_TRUE(torch::equal(expected_ious, ious));
}

TEST(calculate_iou, iou_no_overlap) {
    auto boxes = torch::tensor({1, 2, 3, 4, 10, 20, 20, 30}, {torch::kFloat32});
    boxes = boxes.view({2, 4});
    auto areas = torch::tensor({4, 100}, {torch::kFloat32});
    int idx = 0;
    torch::Tensor ious = calculate_iou(boxes, areas, idx);

    auto expected_ious = torch::tensor({0.0}, {torch::kFloat32});
    ASSERT_TRUE(torch::equal(expected_ious, ious));
}

TEST(update_sorting_order, order_swap) {
    auto boxes = torch::tensor({1, 2, 3, 4, 5, 6, 8, 9}, {torch::kFloat32});
    boxes = boxes.view({2, 4});
    auto areas = torch::tensor({4, 9}, {torch::kFloat32});
    auto scores = torch::tensor({0.7, 0.8}, {torch::kFloat32});

    int idx = 0;
    update_sorting_order(boxes, scores, areas, idx);

    auto expected_boxes = torch::tensor({5, 6, 8, 9, 1, 2, 3, 4}, {torch::kFloat32});
    expected_boxes = expected_boxes.view({2, 4});
    auto expected_areas = torch::tensor({9, 4}, {torch::kFloat32});
    auto expected_scores = torch::tensor({0.8, 0.7}, {torch::kFloat32});

    ASSERT_TRUE(torch::equal(expected_boxes, boxes));
    ASSERT_TRUE(torch::equal(expected_areas, areas));
    ASSERT_TRUE(torch::equal(expected_scores, scores));
}

TEST(update_sorting_order, no_swap) {
    auto boxes = torch::tensor({1, 2, 3, 4, 5, 6, 8, 9}, {torch::kFloat32});
    boxes = boxes.view({2, 4});
    auto areas = torch::tensor({4, 9}, {torch::kFloat32});
    auto scores = torch::tensor({0.8, 0.7}, {torch::kFloat32});

    int idx = 0;
    update_sorting_order(boxes, scores, areas, idx);

    auto expected_boxes = torch::tensor({1, 2, 3, 4, 5, 6, 8, 9}, {torch::kFloat32});
    expected_boxes = expected_boxes.view({2, 4});
    auto expected_areas = torch::tensor({4, 9}, {torch::kFloat32});
    auto expected_scores = torch::tensor({0.8, 0.7}, {torch::kFloat32});

    ASSERT_TRUE(torch::equal(expected_boxes, boxes));
    ASSERT_TRUE(torch::equal(expected_areas, areas));
    ASSERT_TRUE(torch::equal(expected_scores, scores));
}

TEST(soft_nms, no_errors) {
    // This test simply checks that we can run the function without errors, but doesn't validate the results
    torch::manual_seed(42);
    auto boxes = 100.0 * torch::rand({100, 4});
    auto scores = torch::rand({100});

    soft_nms(boxes, scores, 0.5, 0.5);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
