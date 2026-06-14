"""
Generate a binary antenna mask for channel H matrix inpainting.

The H matrix has shape (N_tx, N_rx) = (64, 64).
A mask value of 1 means the link is observed; 0 means it is missing.

Modes
-----
random        : n_tx random Tx × n_rx random Rx grid intersections observed
block         : contiguous top-left n_tx × n_rx rectangle observed
random_rows   : n_tx randomly chosen full transmitter rows observed (all Rx)
random_cols   : n_rx randomly chosen full receiver columns observed (all Tx)
periodic      : every <step>-th Tx and every <step>-th Rx observed
random_entries: ratio fraction of individual (tx, rx) entries chosen at random

Usage examples
--------------
python generate_channel_mask.py --mode random        --n_tx 16 --n_rx 16
python generate_channel_mask.py --mode block         --n_tx 16 --n_rx 16
python generate_channel_mask.py --mode random_rows   --n_tx 16
python generate_channel_mask.py --mode random_cols   --n_rx 16
python generate_channel_mask.py --mode periodic      --step 4
python generate_channel_mask.py --mode random_entries --ratio 0.25
"""

import argparse
import numpy as np
import os


def generate_mask(mode, n_tx=16, n_rx=16, img_size=64, step=4, ratio=0.25, seed=0):
    rng = np.random.default_rng(seed)
    mask = np.zeros((img_size, img_size), dtype=np.float32)

    if mode == 'random':
        tx_idx = rng.choice(img_size, n_tx, replace=False)
        rx_idx = rng.choice(img_size, n_rx, replace=False)
        mask[np.ix_(tx_idx, rx_idx)] = 1.0

    elif mode == 'block':
        tx_idx = np.arange(n_tx)
        rx_idx = np.arange(n_rx)
        mask[np.ix_(tx_idx, rx_idx)] = 1.0

    elif mode == 'random_rows':
        # Full rows: all Rx links are known for n_tx randomly chosen Tx antennas
        tx_idx = rng.choice(img_size, n_tx, replace=False)
        mask[tx_idx, :] = 1.0

    elif mode == 'random_cols':
        # Full columns: all Tx links are known for n_rx randomly chosen Rx antennas
        rx_idx = rng.choice(img_size, n_rx, replace=False)
        mask[:, rx_idx] = 1.0

    elif mode == 'periodic':
        # Every <step>-th Tx and every <step>-th Rx are observed
        tx_idx = np.arange(0, img_size, step)
        rx_idx = np.arange(0, img_size, step)
        mask[np.ix_(tx_idx, rx_idx)] = 1.0

    elif mode == 'random_entries':
        # Random individual (tx, rx) entries; no row/column structure
        flat = rng.choice(img_size * img_size,
                          size=round(ratio * img_size * img_size),
                          replace=False)
        mask.flat[flat] = 1.0

    else:
        raise ValueError(
            f"Unknown mode '{mode}'. "
            "Choose: random | block | random_rows | random_cols | periodic | random_entries"
        )

    return mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',   type=str, default='random',
                        help='Masking strategy (see module docstring)')
    parser.add_argument('--n_tx',  type=int, default=16,
                        help='Number of observed Tx antennas (random / block / random_rows)')
    parser.add_argument('--n_rx',  type=int, default=16,
                        help='Number of observed Rx antennas (random / block / random_cols)')
    parser.add_argument('--step',  type=int, default=4,
                        help='Subsampling step (periodic mode)')
    parser.add_argument('--ratio', type=float, default=0.25,
                        help='Observation ratio 0–1 (random_entries mode)')
    parser.add_argument('--size',  type=int, default=64,
                        help='H matrix dimension (default: 64)')
    parser.add_argument('--seed',  type=int, default=0)
    parser.add_argument('--out',   type=str,
                        default=os.path.join(os.path.dirname(__file__), 'mask.npy'),
                        help='Output .npy path')
    args = parser.parse_args()

    mask = generate_mask(
        mode=args.mode,
        n_tx=args.n_tx,
        n_rx=args.n_rx,
        img_size=args.size,
        step=args.step,
        ratio=args.ratio,
        seed=args.seed,
    )
    np.save(args.out, mask)

    obs_ratio = mask.mean() * 100
    print(f"Mask saved to  : {args.out}")
    print(f"Mode           : {args.mode}")
    print(f"Observed entries: {int(mask.sum())} / {args.size**2}  ({obs_ratio:.1f}%)")