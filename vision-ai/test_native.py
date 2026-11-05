"""Lightweight local checks. Run: python -m unittest discover -s vision-ai -p 'test_native.py'."""
import unittest

import torch

from hawkx_native import HAWKXDetector, decode_detections, detection_loss, targets_from_boxes


class NativeDetectorTest(unittest.TestCase):
    def test_tensor_shape_and_backprop(self):
        model = HAWKXDetector()
        pixels = torch.rand(2, 3, 128, 128)
        logits = model(pixels)
        self.assertEqual(tuple(logits.shape), (2, 5, 8, 8))
        targets = torch.stack([
            targets_from_boxes([[0.2, 0.3, 0.5, 0.8]]),
            targets_from_boxes([]),
        ])
        loss = detection_loss(logits, targets)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(model.network[0].weight.grad)

    def test_empty_decode(self):
        logits = torch.full((5, 8, 8), -20.0)
        self.assertEqual(decode_detections(logits, 128, 128), [])

    def test_bad_box(self):
        with self.assertRaises(ValueError):
            targets_from_boxes([[0.9, 0.2, 0.1, 0.5]])


if __name__ == "__main__":
    unittest.main()
