import os
import numpy as np
import torch
from torch.utils.data import Dataset


def _load_real_imag(path):
    """Load a channel .npy and return a (2, H, W) float32 array [real, imag]."""
    data = np.load(path)
    if np.iscomplexobj(data):
        # (H, W) complex → (2, H, W) float
        data = np.stack([data.real, data.imag], axis=0).astype(np.float32)
    else:
        data = data.astype(np.float32)
        if data.ndim == 2:
            # Single real plane, pad imaginary with zeros
            data = np.stack([data, np.zeros_like(data)], axis=0)
        # else: already (2, H, W)
    return data


def _list_npy(root):
    files = sorted([
        f for f in os.listdir(root)
        if f.endswith('.npy') and f != 'stats.npy'
    ])
    if len(files) == 0:
        raise RuntimeError(f"No .npy files found in '{root}'")
    return files


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
        self.files = _list_npy(root)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = _load_real_imag(os.path.join(self.root, self.files[idx]))
        return torch.from_numpy(data), 0  # 0 = dummy class label


class ChannelDenoiseDataset(Dataset):
    """
    Paired (clean, noisy) channel dataset for DDNM denoising.

    Returns (clean_tensor, noisy_tensor, 0) where BOTH tensors live in the
    diffusion model's normalized clean space [-1, 1] (the model was trained on
    clean channels normalized with the clean min/max c_min/c_max).

    Two input formats are supported via `prenorm`:

      prenorm=False (raw physical values stored):
        Both clean and noisy are min-max normalized with the CLEAN stats:
            v_model = 2 * (v - c_min) / (c_max - c_min) - 1
        Noisy values may fall outside [-1, 1] (its physical range is wider);
        that is expected and harmless for DDNM (y only enters the data term).

      prenorm=True (each file already in [-1, 1] by its OWN stats):
        clean is left as-is (already in model space);
        noisy is remapped noisy-space -> physical -> clean-space:
            v_phys  = (v_noisy + 1) / 2 * (n_max - n_min) + n_min
            v_model = 2 * (v_phys - c_min) / (c_max - c_min) - 1
    """

    def __init__(self, clean_root, noisy_root, c_min, c_max,
                 n_min=None, n_max=None, prenorm=False):
        self.clean_root = clean_root
        self.noisy_root = noisy_root
        self.clean_files = _list_npy(clean_root)
        self.noisy_files = _list_npy(noisy_root)
        if len(self.clean_files) != len(self.noisy_files):
            raise RuntimeError(
                f"clean/noisy count mismatch: {len(self.clean_files)} clean "
                f"vs {len(self.noisy_files)} noisy (paired by sorted filename order)"
            )
        self.c_min = float(c_min)
        self.c_max = float(c_max)
        self.n_min = None if n_min is None else float(n_min)
        self.n_max = None if n_max is None else float(n_max)
        self.prenorm = prenorm
        if self.prenorm and (self.n_min is None or self.n_max is None):
            raise ValueError("prenorm=True requires n_min and n_max (noisy stats)")

    def __len__(self):
        return len(self.clean_files)

    def _norm_clean(self, v):
        return 2.0 * (v - self.c_min) / (self.c_max - self.c_min) - 1.0

    def _noisy_to_model(self, v_noisy):
        if self.prenorm:
            # noisy [-1,1] (noisy stats) -> physical -> clean-normalized [-1,1]
            v_phys = (v_noisy + 1.0) / 2.0 * (self.n_max - self.n_min) + self.n_min
        else:
            # noisy is raw physical
            v_phys = v_noisy
        return self._norm_clean(v_phys)

    def __getitem__(self, idx):
        clean = _load_real_imag(os.path.join(self.clean_root, self.clean_files[idx]))
        noisy = _load_real_imag(os.path.join(self.noisy_root, self.noisy_files[idx]))

        if self.prenorm:
            clean_model = clean  # already in model space
        else:
            clean_model = self._norm_clean(clean)
        noisy_model = self._noisy_to_model(noisy)

        # Per-sample noise level in the model's [-1, 1] normalized scale, i.e.
        # the RMS of (noisy - clean) over all real/imag elements. This equals the
        # global value from scripts/estimate_sigma.py but computed for THIS sample,
        # so DDNM can denoise each sample to its own SNR instead of a fixed sigma_y.
        residual = noisy_model.astype(np.float64) - clean_model.astype(np.float64)
        sigma_y = float(np.sqrt(np.mean(residual ** 2)))

        return (
            torch.from_numpy(clean_model.astype(np.float32)),
            torch.from_numpy(noisy_model.astype(np.float32)),
            sigma_y,
            0,
        )