#include <torch/torch.h>
#include "soft_nms.h"

#ifdef _WIN32
#include <Python.h>
PyMODINIT_FUNC PyInit_soft_nms(void) {
    return NULL;
}
#endif

TORCH_LIBRARY(soft_nms, m) {
    m.def("soft_nms", &soft_nms);
}
