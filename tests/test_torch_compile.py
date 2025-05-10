import os
import unittest

import torch

from pt_soft_nms import batched_soft_nms, soft_nms


class TestCompileSoftNMS(unittest.TestCase):
    def setUp(self):
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        self.boxes = torch.tensor([[20, 20, 40, 40], [10, 10, 20, 20], [20, 20, 35, 35]]).cpu().float()
        self.scores = torch.tensor([0.5, 0.9, 0.11]).cpu().float()
        self.sigma = 0.5
        self.threshold = 0.1

    @unittest.skipIf(os.name == "nt", "Running on Windows")
    def test_compile_cpu(self):
        compiled_soft_nms = torch.compile(soft_nms, fullgraph=True)
        _, keep = compiled_soft_nms(self.boxes, self.scores, self.sigma, self.threshold)
        self.assertTrue(torch.allclose(keep, torch.tensor([1, 0])))

    @unittest.skipIf(
        not torch.cuda.is_available() or os.name == "nt",
        "CUDA not available or running on Windows",
    )
    def test_compile_cuda(self):
        compiled_soft_nms = torch.compile(soft_nms, fullgraph=True)
        _, keep = compiled_soft_nms(self.boxes.cuda(), self.scores.cuda(), self.sigma, self.threshold)
        self.assertTrue(torch.allclose(keep, torch.tensor([1, 0]).cuda()))


class TestCompileBatchedSoftNMS(unittest.TestCase):
    def setUp(self):
        self.boxes = torch.tensor([[20, 20, 40, 40], [10, 10, 20, 20], [20, 20, 35, 35], [20, 20, 35, 35]]).cpu().float()
        self.scores = torch.tensor([0.5, 0.9, 0.11, 0.11]).cpu().float()
        self.idxs = torch.tensor([0, 0, 0, 1]).cpu()
        self.sigma = 0.5
        self.threshold = 0.1

    @unittest.skip("TODO: fix compile for batched_soft_nms")
    def test_compile_cpu(self):
        compiled_soft_nms = torch.compile(batched_soft_nms)
        _ = compiled_soft_nms(self.boxes, self.scores, self.idxs, self.sigma, self.threshold)

    @unittest.skip("TODO: fix compile for batched_soft_nms")
    def test_compile_cuda(self):
        compiled_soft_nms = torch.compile(batched_soft_nms)
        _ = compiled_soft_nms(self.boxes.cuda(), self.scores.cuda(), self.idxs.cuda(), self.sigma, self.threshold)


if __name__ == "__main__":
    unittest.main()
