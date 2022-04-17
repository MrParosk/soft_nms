import os
import unittest
from copy import deepcopy
from tempfile import TemporaryDirectory

import torch

from pt_soft_nms import batched_soft_nms, soft_nms


class TestSoftNMS(unittest.TestCase):
    def setUp(self):
        self.boxes = torch.tensor([[20, 20, 40, 40], [10, 10, 20, 20], [20, 20, 35, 35]]).cpu().float()
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
        self.boxes = torch.tensor([[20, 20, 40, 40], [10, 10, 20, 20], [20, 20, 35, 35], [20, 20, 35, 35]]).cpu().float()
        self.scores = torch.tensor([0.5, 0.9, 0.11, 0.11]).cpu().float()
        self.idxs = torch.tensor([0, 0, 0, 1]).cpu()

    def test_batched_soft_nms(self):
        keep = batched_soft_nms(self.boxes, self.scores, self.idxs, 0.5, 0.1)
        self.assertTrue(torch.allclose(keep, torch.tensor([1, 0, 3])))


class TestingModule(torch.nn.Module):
    def forward(self, boxes, scores):
        return soft_nms(boxes, scores, 0.5, 0.05)


class TestScritable(unittest.TestCase):
    def setUp(self):
        self.module = TestingModule()

        torch.random.manual_seed(42)
        self.boxes = 100 * torch.rand((10, 4))
        self.scores = torch.rand((10))

    def _call_module(self, module):
        _ = module(self.boxes, self.scores)

    def _jit_save_load(self, m):
        jit_module = torch.jit.script(m)
        self._call_module(jit_module)

        with TemporaryDirectory() as dir:
            file = os.path.join(dir, "module.pt")
            torch.jit.save(jit_module, file)
            loaded_module = torch.jit.load(file)

        self._call_module(loaded_module)

    def test_scriptable_cpu(self):
        m = deepcopy(self.module).cpu()
        self._jit_save_load(m)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_scriptable_gpu(self):
        m = deepcopy(self.module).cuda()
        self._jit_save_load(m)


class TestTracing(unittest.TestCase):
    def setUp(self):
        self.module = TestingModule()

        torch.random.manual_seed(42)
        self.boxes = 100 * torch.rand((10, 4))
        self.scores = torch.rand((10))

    def _call_module(self, module):
        _ = module(self.boxes, self.scores)

    def _trace_save_load(self, m):
        trace_module = torch.jit.trace(m, (self.boxes, self.scores))
        self._call_module(trace_module)

        with TemporaryDirectory() as dir:
            file = os.path.join(dir, "module.pt")
            torch.jit.save(trace_module, file)
            loaded_module = torch.jit.load(file)

        self._call_module(loaded_module)

    def test_tracing_cpu(self):
        m = deepcopy(self.module).cpu()
        self._trace_save_load(m)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_tracing_gpu(self):
        m = deepcopy(self.module).cuda()
        self._trace_save_load(m)


if __name__ == "__main__":
    unittest.main()
