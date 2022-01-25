#include <gtest/gtest.h>
#include "../../pt_soft_nms/csrc/soft_nms.h"


TEST(calculate_area, correct_area) {    
    torch::Tensor boxes = torch::tensor({1, 2, 3, 4, 5, 6, 8, 9}, {torch::kFloat32});
    boxes = boxes.view({2, 4});
    torch::Tensor areas = calculate_area(boxes);

    torch::Tensor expected_areas = torch::tensor({4, 9}, {torch::kFloat32});

    ASSERT_TRUE(torch::equal(expected_areas, areas));
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
