"""Read-only ShapeNet HDF5 access; dataset files are never modified."""

from bisect import bisect_right
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ShapeNetDataset(Dataset):
    """Load one point cloud at a time from ``{split}*.h5`` shards.

    Shards contain ``data`` of shape (samples, points, 3) and ``label``
    with one category ID per sample. Optional id2name/id2file JSON files are
    not needed for training or evaluation. No sampling or normalization is
    applied: the first ``num_points`` points of each prepared sample are used.
    """

    def __init__(self, root, split="train", num_points=2048):
        self.root = Path(root).expanduser()
        self.split = split
        self.num_points = int(num_points)
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if self.num_points <= 0:
            raise ValueError("num_points must be positive")
        self.files = sorted(self.root.glob(f"{split}*.h5"))
        if not self.files:
            raise FileNotFoundError(f"No {split}*.h5 shards found in {self.root}")
        self._ends = []
        total = 0
        for path in self.files:
            with h5py.File(path, "r") as shard:
                if "data" not in shard or "label" not in shard:
                    raise ValueError(f"{path} must contain data and label datasets")
                shape = shard["data"].shape
                if len(shape) != 3 or shape[2] != 3 or shape[1] < self.num_points:
                    raise ValueError(
                        f"{path}: expected data shape (N, >= {self.num_points}, 3), got {shape}"
                    )
                label_shape = shard["label"].shape
                if label_shape not in {(shape[0],), (shape[0], 1)}:
                    raise ValueError(f"{path}: expected one label per point cloud")
                total += shape[0]
                self._ends.append(total)
        if total == 0:
            raise ValueError(f"No samples in {self.root} ({split})")

    def __len__(self):
        return self._ends[-1]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect_right(self._ends, index)
        offset = self._ends[shard_index - 1] if shard_index else 0
        # Per-item handles remain safe across DataLoader workers and leave no
        # writable handles or entire-dataset copies in memory.
        with h5py.File(self.files[shard_index], "r") as shard:
            points = np.asarray(shard["data"][index - offset, :self.num_points], dtype=np.float32)
            label = int(np.asarray(shard["label"][index - offset]).reshape(-1)[0])
        if not np.isfinite(points).all():
            raise ValueError(f"Non-finite coordinates at sample {index}")
        return torch.from_numpy(points.copy()), label
