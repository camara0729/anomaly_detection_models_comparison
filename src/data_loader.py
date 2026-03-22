"""Data loading utilities for EGB-250 .mat (HDF5 v7.3) files.

Dataset layout
--------------
Each file contains ``data/Analog50k`` of shape (500000, 9):
  - All 9 sensor columns (0-indexed) per EP_011V0 Table 5:
      MC1:  Microphone 1, 1st Stage valves Vertical   (col 0)
      MC2:  Microphone 2, 2nd Stage valves Vertical   (col 1)
      CVC1: Current Clamp 1, Power line 1             (col 2)
      CVC2: Current Clamp 2, Power line 2             (col 3)
      CVC3: Current Clamp 3, Power line 3             (col 4)
      A1:   1S_IV Vertical                            (col 5)
      A2:   2S_DV Vertical                            (col 6)
      A3:   B1 Radial Vertical — bearing under study  (col 7)
      A4:   B2 Radial Horizontal                      (col 8)

Returns arrays shaped (9, N) — channels-first convention.
"""

import glob
import os
import re
from pathlib import Path
from typing import List

import h5py
import numpy as np

_DATASET_PATH = "data/Analog50k"
_SENSOR_COLS = slice(0, 9)  # all 9 sensor columns (0-indexed)

SENSOR_NAMES: List[str] = ["MC1", "MC2", "CVC1", "CVC2", "CVC3", "A1", "A2", "A3", "A4"]


def load_mat_sensors(file_path: str) -> np.ndarray:
    """Load all 9 sensor channels from a single .mat (HDF5 v7.3) file.

    Parameters
    ----------
    file_path : str
        Path to the .mat file.

    Returns
    -------
    np.ndarray
        Array of shape (9, N) and dtype float32.
        Rows correspond to SENSOR_NAMES: MC1, MC2, CVC1, CVC2, CVC3, A1, A2, A3, A4.

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with h5py.File(path, "r") as f:
        data = f[_DATASET_PATH][:, _SENSOR_COLS]  # (N, 9)

    return data.T.astype(np.float32)  # (9, N)


def list_mat_files(data_dir: str) -> List[str]:
    """Return a sorted list of all .mat files under *data_dir*.

    Searches recursively inside ``P1/``, ``P2/``, ``P3/``, ``P4/``
    subdirectories.

    Parameters
    ----------
    data_dir : str
        Root data directory (e.g. ``"data"``).

    Returns
    -------
    List[str]
        Sorted list of absolute-or-relative file paths.
    """
    pattern = os.path.join(data_dir, "P[1-4]", "*.mat")
    files = glob.glob(pattern)
    return sorted(files)


def load_all_runs(class_dir: str) -> List[np.ndarray]:
    """Load all runs for a single fault class directory.

    Parameters
    ----------
    class_dir : str
        Directory containing the .mat files for one class (e.g. ``"data/P1"``).

    Returns
    -------
    List[np.ndarray]
        List of arrays, each of shape (9, N) and dtype float32, sorted by
        run filename.

    Raises
    ------
    FileNotFoundError
        If *class_dir* does not exist.
    ValueError
        If no .mat files are found in *class_dir*.
    """
    path = Path(class_dir)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {class_dir}")

    files = sorted(path.glob("*.mat"), key=lambda p: int(re.search(r"R(\d+)", p.stem).group(1)))
    if not files:
        raise ValueError(f"No .mat files found in: {class_dir}")

    return [load_mat_sensors(str(f)) for f in files]
