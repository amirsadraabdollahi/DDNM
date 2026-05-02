import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ChannelHDataset(Dataset):
    """
    Dataset for complex channel H matrices stored as .npy files.

    Accepted file shapes per sample:
      (H, W) complex  →  stacked to (2, H, W) float  [real, imag]
      (2, H, W) float →  used as-is                  [real, imag]

    Values must already be normalized to [-1, 1] (matching the diffusion
    model's training distribution). Use d_min / d_max in the config for
    denormalization at inference time.
    """

    def __init__(self, root):
        self.root = root
        self.files = sorted([
            f for f in os.listdir(root)
            if f.endswith('.npy') and f != 'stats.npy'
        ])
        if len(self.files) == 0:
            raise RuntimeError(f"No .npy files found in '{root}'")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.root, self.files[idx]))

        if np.iscomplexobj(data):
            # (H, W) complex → (2, H, W) float
            data = np.stack([data.real, data.imag], axis=0).astype(np.float32)
        else:
            data = data.astype(np.float32)
            if data.ndim == 2:
                # Single real plane, pad imaginary with zeros
                data = np.stack([data, np.zeros_like(data)], axis=0)
            # else: already (2, H, W)

        return torch.from_numpy(data), 0  # 0 = dummy class label