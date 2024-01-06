import os
import unittest

import torch

from pt_soft_nms import batched_soft_nms, soft_nms


def _under_version_two():
    return torch.__version__ < (2, 0)


class TestCompileSoftNMS(unittest.TestCase):
    def setUp(self):
        self.boxes = torch.tensor([[20, 20, 40, 40], [10, 10, 20, 20], [20, 20, 35, 35]]).cpu().float()
        self.scores = torch.tensor([0.5, 0.9, 0.11]).cpu().float()
        self.sigma = 0.5
        self.threshold = 0.1

    @unittest.skipIf(_under_version_two() or os.name == "nt", "Torch version < 2.0 or running on Windows")
    def test_compile_cpu(self):
        compiled_soft_nms = torch.compile(soft_nms)
        _ = compiled_soft_nms(self.boxes, self.scores, self.sigma, self.threshold)

    @unittest.skipIf(
        _under_version_two() or not torch.cuda.is_available() or os.name == "nt",
        "Torch version < 2.0 or CUDA not available or running on Windows",
    )
    def test_compile_cuda(self):
        compiled_soft_nms = torch.compile(soft_nms)
        _ = compiled_soft_nms(self.boxes.cuda(), self.scores.cuda(), self.sigma, self.threshold)


class TestCompileBatchedSoftNMS(unittest.TestCase):
    def setUp(self):
        self.boxes = torch.tensor([[20, 20, 40, 40], [10, 10, 20, 20], [20, 20, 35, 35], [20, 20, 35, 35]]).cpu().float()
        self.scores = torch.tensor([0.5, 0.9, 0.11, 0.11]).cpu().float()
        self.idxs = torch.tensor([0, 0, 0, 1]).cpu()
        self.sigma = 0.5
        self.threshold = 0.1

    @unittest.skipIf(_under_version_two() or os.name == "nt", "Torch version < 2.0 or running on Windows")
    def test_compile_cpu(self):
        compiled_soft_nms = torch.compile(batched_soft_nms)
        _ = compiled_soft_nms(self.boxes, self.scores, self.idxs, self.sigma, self.threshold)

    @unittest.skipIf(
        _under_version_two() or not torch.cuda.is_available() or os.name == "nt",
        "Torch version < 2.0 or CUDA not available or running on Windows",
    )
    def test_compile_cuda(self):
        compiled_soft_nms = torch.compile(batched_soft_nms)
        _ = compiled_soft_nms(self.boxes.cuda(), self.scores.cuda(), self.idxs.cuda(), self.sigma, self.threshold)


if __name__ == "__main__":
    unittest.main()
