import unittest

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


if __name__ == "__main__":
    unittest.main()
