import unittest
import torch
from copy import deepcopy
from pt_soft_nms import soft_nms


class TestCorrectImplemented(unittest.TestCase):
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

    def test_gpu(self):
        _ = soft_nms(self.boxes.cuda(), self.scores.cuda(), 0.5, 0.1)


class TestScritable(unittest.TestCase):
    def setUp(self):
        class TestingModule(torch.nn.Module):
            def forward(self, boxes, scores):
                return soft_nms(boxes, scores, 0.5, 0.05)

        self.module = TestingModule()

    def test_scriptable_cpu(self):
        m = deepcopy(self.module).cpu()
        _ = torch.jit.script(m)

    def test_scriptable_gpu(self):
        m = deepcopy(self.module).cuda()
        _ = torch.jit.script(m)


if __name__ == "__main__":
    unittest.main()
