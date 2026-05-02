"""
Generate a binary antenna mask for channel H matrix inpainting.

The H matrix has shape (N_tx, N_rx) = (64, 64).
A mask value of 1 means the link is observed; 0 means it is missing.

The mask is saved as  exp/inp_masks/mask.npy  with shape (64, 64) so it
broadcasts correctly across the 2-channel (real/imag) input tensors.

Usage examples
--------------
# Observe 16 random transmitters and 16 random receivers:
python generate_channel_mask.py --mode random --n_tx 16 --n_rx 16

# Observe a contiguous block of transmitters 0-15 and receivers 0-15:
python generate_channel_mask.py --mode block --n_tx 16 --n_rx 16
"""

import argparse
import numpy as np
import os

def generate_mask(mode, n_tx, n_rx, img_size=64, seed=0):
    rng = np.random.default_rng(seed)
    mask = np.zeros((img_size, img_size), dtype=np.float32)

    if mode == 'random':
        tx_idx = rng.choice(img_size, n_tx, replace=False)
        rx_idx = rng.choice(img_size, n_rx, replace=False)
    elif mode == 'block':
        tx_idx = np.arange(n_tx)
        rx_idx = np.arange(n_rx)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'random' or 'block'.")

    mask[np.ix_(tx_idx, rx_idx)] = 1.0
    return mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',    type=str, default='random',
                        help="'random' (random subset) or 'block' (first n_tx/n_rx)")
    parser.add_argument('--n_tx',   type=int, default=16,
                        help='Number of observed transmit antennas')
    parser.add_argument('--n_rx',   type=int, default=16,
                        help='Number of observed receive antennas')
    parser.add_argument('--size',   type=int, default=64,
                        help='H matrix dimension (default: 64)')
    parser.add_argument('--seed',   type=int, default=0)
    parser.add_argument('--out',    type=str,
                        default=os.path.join(os.path.dirname(__file__), 'mask.npy'))
    args = parser.parse_args()

    mask = generate_mask(args.mode, args.n_tx, args.n_rx, args.size, args.seed)
    np.save(args.out, mask)

    obs_ratio = mask.mean() * 100
    print(f"Mask saved to {args.out}")
    print(f"Observed entries: {int(mask.sum())} / {args.size**2}  ({obs_ratio:.1f}%)")