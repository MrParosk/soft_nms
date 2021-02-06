#include <torch/torch.h>
#include "soft_nms/soft_nms.h"

TORCH_LIBRARY(ts_ops, m)
{
    m.def("soft_nms", &soft_nms);
}
