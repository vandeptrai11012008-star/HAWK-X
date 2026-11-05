"""Offline tests for dataset metadata validation (no ML dependencies needed)."""
import tempfile
import unittest
from pathlib import Path

from train import check_dataset, write_config


class DatasetCheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "person-dataset"
        for split in ("train", "val"):
            (self.root / "images" / split).mkdir(parents=True)
            (self.root / "labels" / split).mkdir(parents=True)
            # The dataset validator checks paths and labels, not image decodability.
            (self.root / "images" / split / "person.jpg").write_bytes(b"test-image")
            (self.root / "labels" / split / "person.txt").write_text(
                "0 0.5 0.5 0.2 0.5\n", encoding="utf-8"
            )

    def test_valid_person_dataset(self):
        path, counts = check_dataset(self.root)
        self.assertEqual(path, self.root)
        self.assertEqual(counts["train"], (1, 1))
        self.assertEqual(counts["val"], (1, 1))

    def test_rejects_invalid_class(self):
        (self.root / "labels" / "train" / "person.txt").write_text(
            "1 0.5 0.5 0.2 0.5\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "expected"):
            check_dataset(self.root)

    def test_rejects_outside_box(self):
        (self.root / "labels" / "train" / "person.txt").write_text(
            "0 0.99 0.5 0.5 0.4\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            check_dataset(self.root)

    def test_missing_labels_are_valid_for_empty_images(self):
        (self.root / "images" / "val" / "background.jpg").write_bytes(b"test-image")
        _, counts = check_dataset(self.root)
        self.assertEqual(counts["val"], (2, 1))


if __name__ == "__main__":
    unittest.main()
