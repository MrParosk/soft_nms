import unittest
import os
import torch
from copy import deepcopy
from tempfile import TemporaryDirectory
from pt_soft_nms import soft_nms, batched_soft_nms


class TestSoftNMS(unittest.TestCase):
    def setUp(self):
        self.boxes = torch.tensor(
            [[20, 20, 40, 40], [10, 10, 20, 20], [20, 20, 35, 35]]).cpu().float()
        self.scores = torch.tensor([0.5, 0.9, 0.11]).cpu().float()

    def test_correct_keep(self):
        _, keep = soft_nms(self.boxes, self.scores, 0.5, 0.1)
        self.assertTrue(torch.allclose(keep, torch.tensor([1, 0])))

    def test_indexing_works(self):
        # Making sure that we can use keep for indexing
        _, keep = soft_nms(self.boxes, self.scores, 0.5, 0.1)
        _ = self.boxes[keep, :]

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_gpu(self):
        _ = soft_nms(self.boxes.cuda(), self.scores.cuda(), 0.5, 0.1)


class TestBatchSoftNMS(unittest.TestCase):
    def setUp(self):
        # Overlap but with different class-idx
        self.boxes = torch.tensor(
            [[20, 20, 40, 40], [10, 10, 20, 20], [20, 20, 35, 35], [20, 20, 35, 35]]).cpu().float()
        self.scores = torch.tensor([0.5, 0.9, 0.11, 0.11]).cpu().float()
        self.idxs = torch.tensor([0, 0, 0, 1]).cpu()

    def test_batched_soft_nms(self):
        keep = batched_soft_nms(self.boxes, self.scores, self.idxs, 0.5, 0.1)
        self.assertTrue(torch.allclose(keep, torch.tensor([1, 0, 3])))


class TestScritable(unittest.TestCase):
    def setUp(self):
        class TestingModule(torch.nn.Module):
            def forward(self, boxes, scores):
                return soft_nms(boxes, scores, 0.5, 0.05)

        self.module = TestingModule()

    def jit_save_load(self, m):
        jit_mod = torch.jit.script(m)
        with TemporaryDirectory() as dir:
            file = os.path.join(dir, "module.pt")
            torch.jit.save(jit_mod, file)
            torch.jit.load(file)

    def test_scriptable_cpu(self):
        m = deepcopy(self.module).cpu()
        self.jit_save_load(m)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_scriptable_gpu(self):
        m = deepcopy(self.module).cuda()
        self.jit_save_load(m)


if __name__ == "__main__":
    unittest.main()
