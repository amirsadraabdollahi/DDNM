#!/usr/bin/env python3
"""
Estimate the DDNM denoising noise level (sigma_y) from paired raw clean/noisy
channel samples, expressed in the [-1, 1] CLEAN-normalized scale that the
diffusion model works in.

Usage:
    python scripts/estimate_sigma.py \
        --clean exp/datasets/channel_denoise/clean \
        --noisy exp/datasets/channel_denoise/noisy \
        --c_min -3.19540334 --c_max 3.15650487

Paste the printed value into `data.sigma_y` of configs/channel_denoise.yml.
"""
import argparse
import os

import numpy as np


def load_real_imag(path):
    data = np.load(path)
    if np.iscomplexobj(data):
        data = np.stack([data.real, data.imag], axis=0).astype(np.float64)
    else:
        data = data.astype(np.float64)
        if data.ndim == 2:
            data = np.stack([data, np.zeros_like(data)], axis=0)
    return data


def list_npy(root):
    files = sorted(f for f in os.listdir(root)
                   if f.endswith('.npy') and f != 'stats.npy')
    if not files:
        raise RuntimeError(f"No .npy files found in '{root}'")
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clean', required=True, help='dir of raw clean .npy')
    ap.add_argument('--noisy', required=True, help='dir of raw noisy .npy')
    ap.add_argument('--c_min', type=float, required=True, help='clean min')
    ap.add_argument('--c_max', type=float, required=True, help='clean max')
    ap.add_argument('--limit', type=int, default=0,
                    help='use only the first N pairs (0 = all)')
    args = ap.parse_args()

    clean_files = list_npy(args.clean)
    noisy_files = list_npy(args.noisy)
    if len(clean_files) != len(noisy_files):
        raise RuntimeError(
            f"count mismatch: {len(clean_files)} clean vs {len(noisy_files)} noisy")
    n = len(clean_files) if args.limit <= 0 else min(args.limit, len(clean_files))

    sq_sum = 0.0
    count = 0
    for i in range(n):
        c = load_real_imag(os.path.join(args.clean, clean_files[i]))
        z = load_real_imag(os.path.join(args.noisy, noisy_files[i]))
        r = z - c                       # residual noise in physical units
        sq_sum += float(np.sum(r ** 2))
        count += r.size

    sigma_phys = (sq_sum / count) ** 0.5
    span = args.c_max - args.c_min
    sigma_y = 2.0 * sigma_phys / span   # map physical std into the [-1,1] clean scale

    # rough SNR for sanity
    print(f"pairs used:            {n}")
    print(f"residual std (phys):   {sigma_phys:.6f}")
    print(f"clean span (max-min):  {span:.6f}")
    print(f"==> sigma_y (use this in config): {sigma_y:.6f}")


if __name__ == '__main__':
    main()