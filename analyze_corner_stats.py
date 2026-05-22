import argparse
import os
from collections import Counter

import numpy as np


def load_names(dataset_root, split_name):
    split_path = os.path.join(dataset_root, f"{split_name}_list.txt")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Cannot find split file: {split_path}")
    with open(split_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def count_corners(dataset_root, names):
    annot_dir = os.path.join(dataset_root, "annot")
    counts = []
    missing = []

    for name in names:
        annot_path = os.path.join(annot_dir, name + ".npy")
        if not os.path.exists(annot_path):
            missing.append(name)
            continue
        annot = np.load(annot_path, allow_pickle=True, encoding="latin1").tolist()
        counts.append((name, len(annot)))

    return counts, missing


def summarize_split(split_name, counts, limits):
    values = np.array([count for _, count in counts], dtype=np.int32)
    print(f"\n[{split_name}]")
    print(f"usable_samples: {len(values)}")

    if len(values) == 0:
        return

    print(f"min_corners: {int(values.min())}")
    print(f"mean_corners: {values.mean():.2f}")
    print(f"median_corners: {float(np.median(values)):.1f}")
    print(f"p90_corners: {float(np.percentile(values, 90)):.1f}")
    print(f"p95_corners: {float(np.percentile(values, 95)):.1f}")
    print(f"p99_corners: {float(np.percentile(values, 99)):.1f}")
    print(f"max_corners: {int(values.max())}")

    for limit in limits:
        over_mask = values > limit
        over_count = int(over_mask.sum())
        keep_count = int((~over_mask).sum())
        ratio = 100.0 * over_count / len(values)
        print(
            f"limit>{limit}: skipped={over_count} kept={keep_count} skipped_ratio={ratio:.2f}%"
        )

    top_items = Counter(dict(counts)).most_common(10)
    print("top_10_complex_samples:")
    for name, count in top_items:
        print(f"  {name}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Analyze GT corner-count distribution for each split.")
    parser.add_argument("--dataset_root", required=True, type=str,
                        help="Processed dataset root containing annot/ and split txt files")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"],
                        help="Splits to analyze, e.g. train valid test")
    parser.add_argument("--limits", nargs="+", default=[100, 150], type=int,
                        help="Corner limits to report skip ratios for")
    args = parser.parse_args()

    print(f"dataset_root: {args.dataset_root}")
    print(f"limits: {args.limits}")

    total_missing = 0
    for split_name in args.splits:
        names = load_names(args.dataset_root, split_name)
        counts, missing = count_corners(args.dataset_root, names)
        summarize_split(split_name, counts, args.limits)
        if missing:
            total_missing += len(missing)
            print(f"missing_annots: {len(missing)}")
            for name in missing[:10]:
                print(f"  missing: {name}")
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")

    if total_missing == 0:
        print("\nAll requested splits were analyzed successfully.")


if __name__ == "__main__":
    main()
