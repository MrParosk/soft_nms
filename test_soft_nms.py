import unittest
import torch
from copy import deepcopy
from ts_ops import soft_nms


class TestCorrectImplemented(unittest.TestCase):
    def setUp(self):
        self.dets = torch.tensor(
            [[20, 20, 40, 40], [10, 10, 20, 20], [20, 20, 35, 35]]).float()
        self.scores = torch.tensor([0.5, 0.9, 0.11]).float()

    def test_correct_keep(self):
        keep = soft_nms(self.dets, self.scores, 0.5, 0.1)
        self.assertTrue(torch.allclose(keep, torch.tensor([1, 0])))

    def test_indexing_works(self):
        # Making sure that we can use keep for indexing
        keep = soft_nms(self.dets, self.scores, 0.5, 0.1)
        _ = self.dets[keep, :]


class TestScritable(unittest.TestCase):
    def setUp(self):
        class TestingModule(torch.nn.Module):
            def forward(self, dets, scores):
                return torch.ops.ts_ops.soft_nms(dets, scores, 0.5, 0.05)

        self.module = TestingModule()

    def test_scriptable_cpu(self):
        m = deepcopy(self.module).cpu()
        _ = torch.jit.script(m)


if __name__ == "__main__":
    unittest.main()
