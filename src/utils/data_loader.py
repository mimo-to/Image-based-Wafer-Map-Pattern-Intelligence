import pickle
import os
import sys
from collections import Counter
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# Legacy pandas pickle support
if "pandas.indexes" not in sys.modules:
    sys.modules["pandas.indexes"] = pd.core.indexes


def load_wafer_dataset(
    pkl_path: str,
) -> Tuple[List[np.ndarray], List[int], pd.DataFrame]:
    """
    Load and validate WM-811K dataset from pickle file.

    Args:
        pkl_path: Path to LSWMD.pkl file

    Returns:
        Tuple of (images, labels, metadata_df)
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Dataset file not found: {pkl_path}")

    print(f"Loading dataset from {pkl_path}...")
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
    except Exception as e:
        raise ValueError(f"Failed to load pickle: {e}")

    # Define class mapping
    class_map = {
        "none": 0,
        "Center": 1,
        "Donut": 2,
        "Edge-Loc": 3,
        "Edge-Ring": 4,
        "Loc": 5,
        "Near-full": 6,
        "Random": 7,
        "Scratch": 8,
    }

    images: List[np.ndarray] = []
    labels: List[int] = []
    metadata_rows: List[Dict[str, Any]] = []

    skipped_count = 0

    # Optimize iteration for DataFrames
    iterator = data
    total = len(data)

    if isinstance(data, pd.DataFrame):
        iterator = data.itertuples()

    for record in tqdm(iterator, total=total, desc="Processing wafers"):
        try:
            # Extract fields from tuple or dict
            if isinstance(record, tuple) and hasattr(record, "waferMap"):
                wafer_map = record.waferMap
                failure_type = record.failureType
                lot_name = getattr(record, "lotName", "")
                wafer_index = getattr(record, "waferIndex", 0)
            elif isinstance(record, dict):
                wafer_map = record.get("waferMap")
                failure_type = record.get("failureType")
                lot_name = record.get("lotName", "")
                wafer_index = record.get("waferIndex", 0)
            else:
                wafer_map = getattr(record, "waferMap", None)
                failure_type = getattr(record, "failureType", None)
                lot_name = getattr(record, "lotName", "")
                wafer_index = getattr(record, "waferIndex", 0)

            # Skip invalid samples
            if wafer_map is None or failure_type is None:
                skipped_count += 1
                continue

            # Ensure 2D numpy array
            wafer_map = np.array(wafer_map)

            if wafer_map.ndim != 2:
                skipped_count += 1
                continue

            # Normalize failureType format
            if isinstance(failure_type, np.ndarray):
                if failure_type.size == 0:
                    f_label = "none"
                else:
                    item = failure_type.flat[0]
                    f_label = str(item)
            elif isinstance(failure_type, list):
                if len(failure_type) == 0:
                    f_label = "none"
                else:
                    item = failure_type[0]
                    if isinstance(item, list) and len(item) > 0:
                        f_label = str(item[0])
                    else:
                        f_label = str(item)
            elif isinstance(failure_type, str):
                f_label = failure_type
            else:
                skipped_count += 1
                continue

            f_label = f_label.strip()

            # Map to integer label
            if f_label not in class_map:
                skipped_count += 1
                continue

            label_idx = class_map[f_label]

            images.append(wafer_map)
            labels.append(label_idx)
            metadata_rows.append(
                {
                    "lotName": lot_name,
                    "waferIndex": wafer_index,
                    "failureType": f_label,
                    "mapped_label": label_idx,
                }
            )

        except Exception:
            skipped_count += 1
            continue

    # Create DataFrame
    metadata_df = pd.DataFrame(metadata_rows)

    # Verify all classes exist
    unique_labels = sorted(set(labels))
    expected_classes = list(range(9))
    if unique_labels != expected_classes:
        missing = set(expected_classes) - set(unique_labels)
        print(f"WARNING: Datset missing classes: {missing}")

    # Calculate statistics
    valid_count = len(images)

    # Image dimensions
    if valid_count > 0:
        heights = [img.shape[0] for img in images]
        widths = [img.shape[1] for img in images]
        h_min, h_max = min(heights), max(heights)
        w_min, w_max = min(widths), max(widths)
    else:
        h_min = h_max = w_min = w_max = 0

    print("-" * 40)
    print(f"Data Loading Complete: {valid_count} valid samples")
    print(f"Skipped samples: {skipped_count}")
    print("-" * 40)
    print("Class Distribution:")

    # Distribution
    dist = Counter(labels)
    # Reverse map for printing
    inv_map = {v: k for k, v in class_map.items()}

    for lbl, count in sorted(dist.items()):
        name = inv_map.get(lbl, "Unknown")
        pct = (count / valid_count) * 100 if valid_count > 0 else 0
        print(f"  {name} ({lbl}): {count} ({pct:.2f}%)")

    print("-" * 40)
    print(f"Image Heights: Min={h_min}, Max={h_max}")
    print(f"Image Widths:  Min={w_min}, Max={w_max}")
    print("-" * 40)

    return images, labels, metadata_df
