"""
Generate the full curated set of evaluation masks described in MASKS.md.

Run from the repo root:
    python exp/inp_masks/generate_all_masks.py

All masks are saved to exp/inp_masks/.
"""

import os
import sys
import numpy as np

# Allow importing generate_mask from the same directory
sys.path.insert(0, os.path.dirname(__file__))
from generate_channel_mask import generate_mask

OUT_DIR = os.path.dirname(__file__)

CATALOGUE = [
    # ── Group A: density sweep (random grid) ──────────────────────────────────
    ("random_8x8",     dict(mode="random",        n_tx=8,  n_rx=8)),
    ("random_16x16",   dict(mode="random",        n_tx=16, n_rx=16)),
    ("random_32x32",   dict(mode="random",        n_tx=32, n_rx=32)),

    # ── Group B: structure comparison at 25% density ──────────────────────────
    # random_32x32 already covers the random-grid entry for this group
    ("block_32x32",    dict(mode="block",         n_tx=32, n_rx=32)),
    ("rows_16",        dict(mode="random_rows",   n_tx=16)),
    ("cols_16",        dict(mode="random_cols",   n_rx=16)),
    ("periodic_s2",    dict(mode="periodic",      step=2)),
    ("entries_25pct",  dict(mode="random_entries", ratio=0.25)),

    # ── Group C: 50% upper-bound reference ────────────────────────────────────
    ("rows_32",        dict(mode="random_rows",   n_tx=32)),
    ("cols_32",        dict(mode="random_cols",   n_rx=32)),
    ("entries_50pct",  dict(mode="random_entries", ratio=0.50)),
]

IMG_SIZE = 64
SEED     = 0

print(f"{'Filename':<22}  {'Mode':<16}  {'Entries':>7}  {'Density':>8}")
print("-" * 62)

for name, kwargs in CATALOGUE:
    mask = generate_mask(img_size=IMG_SIZE, seed=SEED, **kwargs)
    path = os.path.join(OUT_DIR, f"{name}.npy")
    np.save(path, mask)

    entries = int(mask.sum())
    density = mask.mean() * 100
    print(f"{name + '.npy':<22}  {kwargs['mode']:<16}  {entries:>7}  {density:>7.1f}%")

print("-" * 62)
print(f"Saved {len(CATALOGUE)} masks to {OUT_DIR}")
print("(random_32x32.npy is shared between Group A and Group B)")