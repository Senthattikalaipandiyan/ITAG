"""
6_export_to_pkl.py
==================
Exports trained PyTorch model state dictionary from .pth format to a pickle (.pkl) file.
"""

import os
import pickle
from pathlib import Path
import torch

# Base directory and file paths
BASE_DIR = Path(__file__).resolve().parent
PTH_FILE = BASE_DIR / "indian_games_detector.pth"
PKL_FILE = BASE_DIR / "indian_games_detector.pkl"


def export_weights_to_pkl(pth_path=PTH_FILE, pkl_path=PKL_FILE):
    """Loads weights from .pth file and exports them to .pkl file using pickle."""
    pth_path = Path(pth_path)
    pkl_path = Path(pkl_path)

    if not pth_path.exists():
        raise FileNotFoundError(f"Error: Weights file '{pth_path}' not found!")

    print(f"[1/2] Loading trained weights from: {pth_path.name}...")
    weights = torch.load(str(pth_path), map_location="cpu")

    print(f"[2/2] Exporting weights to: {pkl_path.name} via pickle...")
    with open(pkl_path, "wb") as f:
        pickle.dump(weights, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Confirmation output
    print("=" * 60)
    print(f"[SUCCESS] Successfully created '{pkl_path.name}'!")
    print(f"Saved Path : {pkl_path.resolve()}")
    print(f"File Size  : {os.path.getsize(pkl_path):,} bytes")
    print(f"Keys Count : {len(weights)} layer state tensors")
    print("=" * 60)


if __name__ == "__main__":
    export_weights_to_pkl()
