#include <torch/torch.h>
#include "soft_nms.h"

#ifdef _WIN32
#include <Python.h>
PyMODINIT_FUNC PyInit_soft_nms(void) {
    return NULL;
}
#endif

TORCH_LIBRARY(soft_nms, m) {
    m.def("soft_nms(Tensor boxes, Tensor scores, float sigma, float score_threshold) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(soft_nms, CPU, m) {
    m.impl("soft_nms", &soft_nms);
}

TORCH_LIBRARY_IMPL(soft_nms, CUDA, m) {
    m.impl("soft_nms", &soft_nms);
}
