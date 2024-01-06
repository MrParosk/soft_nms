import os
import unittest
from copy import deepcopy
from tempfile import TemporaryDirectory

import torch

from pt_soft_nms import batched_soft_nms, soft_nms


def _save_load_module(jit_module):
    with TemporaryDirectory() as dir:
        file = os.path.join(dir, "module.pt")
        torch.jit.save(jit_module, file)
        loaded_module = torch.jit.load(file)
    return loaded_module


class TestSoftNMSModule(torch.nn.Module):
    def forward(self, boxes, scores):
        return soft_nms(boxes, scores, 0.5, 0.05)


class TestBatchedSoftNMS(torch.nn.Module):
    def forward(self, boxes, scores, idxs):
        return batched_soft_nms(boxes, scores, idxs, 0.5, 0.05)


class TestSoftNMSScriptable(unittest.TestCase):
    def setUp(self):
        self.module = TestSoftNMSModule()

        torch.random.manual_seed(42)
        self.boxes = 100 * torch.rand((10, 4))
        self.scores = torch.rand((10))

    def _call_module(self, module):
        _ = module(self.boxes, self.scores)

    def _jit_save_load(self, m):
        jit_module = torch.jit.script(m)
        self._call_module(jit_module)

        loaded_module = _save_load_module(jit_module)
        self._call_module(loaded_module)

    def test_scriptable_cpu(self):
        m = deepcopy(self.module).cpu()
        self._jit_save_load(m)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_scriptable_gpu(self):
        m = deepcopy(self.module).cuda()
        self._jit_save_load(m)


class TestBatchedSoftNMSScriptable(unittest.TestCase):
    def setUp(self):
        self.module = TestBatchedSoftNMS()

        torch.random.manual_seed(42)
        self.boxes = 100 * torch.rand((10, 4))
        self.scores = torch.rand((10))
        self.idxs = torch.randint(0, 2, size=(10,))

    def _call_module(self, module):
        _ = module(self.boxes, self.scores, self.idxs)

    def _jit_save_load(self, m):
        jit_module = torch.jit.script(m)
        self._call_module(jit_module)

        loaded_module = _save_load_module(jit_module)
        self._call_module(loaded_module)

    def test_scriptable_cpu(self):
        m = deepcopy(self.module).cpu()
        self._jit_save_load(m)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_scriptable_gpu(self):
        m = deepcopy(self.module).cuda()
        self._jit_save_load(m)


class TestSoftNMSTracing(unittest.TestCase):
    def setUp(self):
        self.module = TestSoftNMSModule()

        torch.random.manual_seed(42)
        self.boxes = 100 * torch.rand((10, 4))
        self.scores = torch.rand((10))

    def _call_module(self, module):
        _ = module(self.boxes, self.scores)

    def _trace_save_load(self, m):
        trace_module = torch.jit.trace(m, (self.boxes, self.scores))
        self._call_module(trace_module)

        loaded_module = _save_load_module(trace_module)
        self._call_module(loaded_module)

    def test_tracing_cpu(self):
        m = deepcopy(self.module).cpu()
        self._trace_save_load(m)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_tracing_gpu(self):
        m = deepcopy(self.module).cuda()
        self._trace_save_load(m)


class TestBatchedSoftNMSTracing(unittest.TestCase):
    def setUp(self):
        self.module = TestBatchedSoftNMS()

        torch.random.manual_seed(42)
        self.boxes = 100 * torch.rand((10, 4))
        self.scores = torch.rand((10))
        self.idxs = torch.randint(0, 2, size=(10,))

    def _call_module(self, module):
        _ = module(self.boxes, self.scores, self.idxs)

    def _trace_save_load(self, m):
        trace_module = torch.jit.trace(m, (self.boxes, self.scores, self.idxs))
        self._call_module(trace_module)

        loaded_module = _save_load_module(trace_module)
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
