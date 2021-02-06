#include <torch/torch.h>
#include "soft_nms.h"

TORCH_LIBRARY(soft_nms, m)
{
    m.def("soft_nms", &soft_nms);
}
