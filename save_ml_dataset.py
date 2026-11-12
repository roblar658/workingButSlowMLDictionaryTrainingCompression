#!/usr/bin/env python3
"""
CLI entry point to discover morphological cover rules and generate ml_dataset.pkl.
Delegates to the modular compressor.morphology subpackage.
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from compressor.morphology.dataset import DatasetBuilder

def main():
    dict_path = os.path.join(HERE, "dict.txt")
    out_path = os.path.join(HERE, "ml_dataset.pkl")

    if not os.path.exists(dict_path):
        print(f"Error: {dict_path} not found.")
        sys.exit(1)

    print(f"Loading dictionary from {dict_path} ...")
    t0 = time.perf_counter()
    builder = DatasetBuilder(dict_path)
    
    print("Partitioning words and solving Exact Cover DP...")
    builder.save_pickle(out_path)
    dt = time.perf_counter() - t0
    
    sz = os.path.getsize(out_path)
    print(f"Saved dataset to {out_path}: {sz:,} bytes in {dt:.2f} s")

if __name__ == "__main__":
    main()
