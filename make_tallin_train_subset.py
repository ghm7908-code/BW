import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a smaller, stratified Tallinn training list from a processed BWFormer root. "
            "The script only writes a new train_list file; it does not delete point clouds or annotations."
        )
    )
    parser.add_argument(
        "--processed_root",
        default="/geogfs1/groups/hkurs/u3666068mgh/Tallin/bwformer_trainval_256",
        help="Processed BWFormer root containing annot/ and train_list.txt.",
    )
    parser.add_argument(
        "--source_list",
        default="",
        help="Source train list. Defaults to <processed_root>/train_list.txt.",
    )
    parser.add_argument(
        "--output_list",
        default="",
        help="Output subset list. Defaults to <processed_root>/train_list_subset_<target_count>.txt.",
    )
    parser.add_argument("--target_count", default=768, type=int, help="Number of samples to keep.")
    parser.add_argument("--seed", default=42, type=int, help="Deterministic sampling seed.")
    parser.add_argument("--min_corners", default=4, type=int, help="Drop samples with fewer corners.")
    parser.add_argument(
        "--max_corners",
        default=128,
        type=int,
        help="Drop very complex samples above this corner count. Set <=0 to disable.",
    )
    parser.add_argument(
        "--max_edges",
        default=0,
        type=int,
        help="Drop samples above this unique edge count. Set <=0 to disable.",
    )
    parser.add_argument(
        "--fractions",
        default="0.25,0.45,0.25,0.05",
        help=(
            "Sampling fractions for simple, medium, complex, hard bins. "
            "Default keeps mostly medium/complex examples while retaining a small hard tail."
        ),
    )
    parser.add_argument("--simple_max", default=12, type=int, help="Max corners for the simple bin.")
    parser.add_argument("--medium_max", default=32, type=int, help="Max corners for the medium bin.")
    parser.add_argument("--complex_max", default=64, type=int, help="Max corners for the complex bin.")
    parser.add_argument(
        "--reference_epoch_minutes",
        default=86.0,
        type=float,
        help="Observed full-list minutes per epoch, used only for speed estimates.",
    )
    parser.add_argument("--planned_epochs", default=650, type=int, help="Used only for speed estimates.")
    parser.add_argument("--write_stats_csv", action="store_true", help="Write per-sample stats CSV.")
    return parser.parse_args()


def read_list(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def write_list(path, names):
    with open(path, "w", encoding="utf-8") as handle:
        for name in names:
            handle.write(name + "\n")


def unique_edges(annot):
    corners = list(annot.keys())
    corner_ids = {tuple(corner): idx for idx, corner in enumerate(corners)}
    edges = set()
    missing = 0
    for corner, connections in annot.items():
        src = corner_ids.get(tuple(corner))
        if src is None:
            missing += 1
            continue
        for other in connections:
            dst = corner_ids.get(tuple(other))
            if dst is None:
                missing += 1
                continue
            if src == dst:
                continue
            edges.add(tuple(sorted((src, dst))))
    return edges, missing


def connected_components(num_nodes, edges):
    if num_nodes == 0:
        return 0
    parent = list(range(num_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)
    return len({find(i) for i in range(num_nodes)})


def sample_stats(processed_root, name):
    annot_path = processed_root / "annot" / "{}.npy".format(name)
    annot = np.load(annot_path, allow_pickle=True, encoding="latin1").tolist()
    corners = np.array(list(annot.keys()), dtype=np.float32)
    edges, missing_refs = unique_edges(annot)
    degrees = np.array([len(v) for v in annot.values()], dtype=np.float32)
    z_range = 0.0
    xy_area = 0.0
    if corners.size > 0 and corners.shape[1] >= 3:
        z_range = float(corners[:, 2].max() - corners[:, 2].min())
    if corners.size > 0 and corners.shape[1] >= 2:
        xy = corners[:, :2]
        wh = xy.max(axis=0) - xy.min(axis=0)
        xy_area = float(wh[0] * wh[1])

    num_corners = int(len(corners))
    num_edges = int(len(edges))
    components = connected_components(num_corners, edges)
    max_degree = int(degrees.max()) if len(degrees) else 0
    mean_degree = float(degrees.mean()) if len(degrees) else 0.0
    complexity = num_edges + 0.5 * num_corners + 0.25 * max_degree + 0.01 * z_range

    return {
        "name": name,
        "num_corners": num_corners,
        "num_edges": num_edges,
        "max_degree": max_degree,
        "mean_degree": mean_degree,
        "components": components,
        "missing_refs": int(missing_refs),
        "z_range": z_range,
        "xy_area": xy_area,
        "complexity": float(complexity),
    }


def bin_name(num_corners, args):
    if num_corners <= args.simple_max:
        return "simple"
    if num_corners <= args.medium_max:
        return "medium"
    if num_corners <= args.complex_max:
        return "complex"
    return "hard"


def parse_fractions(text):
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != 4:
        raise ValueError("--fractions must contain four comma-separated values.")
    total = sum(values)
    if total <= 0:
        raise ValueError("--fractions must sum to a positive value.")
    return [value / total for value in values]


def allocate_counts(target_count, fractions):
    raw = [target_count * value for value in fractions]
    counts = [int(np.floor(value)) for value in raw]
    remainder = target_count - sum(counts)
    order = np.argsort([value - np.floor(value) for value in raw])[::-1]
    for idx in order[:remainder]:
        counts[int(idx)] += 1
    return counts


def choose_subset(rows, args):
    rng = np.random.RandomState(args.seed)
    fractions = parse_fractions(args.fractions)
    bins = {"simple": [], "medium": [], "complex": [], "hard": []}
    for row in rows:
        bins[bin_name(row["num_corners"], args)].append(row)

    target_by_bin = dict(zip(["simple", "medium", "complex", "hard"], allocate_counts(args.target_count, fractions)))
    selected = []
    leftovers = []

    for key in ["simple", "medium", "complex", "hard"]:
        items = bins[key]
        rng.shuffle(items)
        take = min(target_by_bin[key], len(items))
        selected.extend(items[:take])
        leftovers.extend(items[take:])

    remaining = args.target_count - len(selected)
    if remaining > 0 and leftovers:
        rng.shuffle(leftovers)
        selected.extend(leftovers[:remaining])

    selected_names = sorted(row["name"] for row in selected)
    return selected_names, bins, target_by_bin


def summarize(rows):
    if not rows:
        return {}
    keys = ["num_corners", "num_edges", "max_degree", "components", "z_range", "complexity"]
    summary = {"count": len(rows)}
    for key in keys:
        values = np.array([row[key] for row in rows], dtype=np.float32)
        summary[key] = {
            "min": float(values.min()),
            "p25": float(np.percentile(values, 25)),
            "median": float(np.percentile(values, 50)),
            "p75": float(np.percentile(values, 75)),
            "p90": float(np.percentile(values, 90)),
            "max": float(values.max()),
            "mean": float(values.mean()),
        }
    return summary


def main():
    args = parse_args()
    processed_root = Path(args.processed_root)
    source_list = Path(args.source_list) if args.source_list else processed_root / "train_list.txt"
    output_list = (
        Path(args.output_list)
        if args.output_list
        else processed_root / "train_list_subset_{}.txt".format(args.target_count)
    )

    if args.target_count <= 0:
        raise ValueError("--target_count must be positive.")
    if not source_list.exists():
        raise FileNotFoundError("Source list not found: {}".format(source_list))

    names = read_list(source_list)
    rows = []
    dropped = []
    for idx, name in enumerate(names, start=1):
        try:
            row = sample_stats(processed_root, name)
        except Exception as exc:
            dropped.append({"name": name, "reason": "load_failed: {}".format(exc)})
            continue

        reason = None
        if row["num_corners"] < args.min_corners:
            reason = "too_few_corners"
        elif args.max_corners > 0 and row["num_corners"] > args.max_corners:
            reason = "too_many_corners"
        elif args.max_edges > 0 and row["num_edges"] > args.max_edges:
            reason = "too_many_edges"
        elif row["num_edges"] < 3:
            reason = "too_few_edges"
        elif row["missing_refs"] > 0:
            reason = "missing_edge_reference"

        if reason is None:
            rows.append(row)
        else:
            item = dict(row)
            item["reason"] = reason
            dropped.append(item)

        if idx % 1000 == 0 or idx == len(names):
            print("scanned {}/{} samples".format(idx, len(names)))

    if len(rows) < args.target_count:
        print(
            "Requested {} samples but only {} passed filters; using all filtered samples.".format(
                args.target_count,
                len(rows),
            )
        )
        args.target_count = len(rows)

    selected_names, bins, target_by_bin = choose_subset(rows, args)
    selected_set = set(selected_names)
    selected_rows = [row for row in rows if row["name"] in selected_set]

    output_list.parent.mkdir(parents=True, exist_ok=True)
    write_list(output_list, selected_names)

    report_path = output_list.with_suffix(".report.json")
    stats_csv_path = output_list.with_suffix(".stats.csv")
    dropped_path = output_list.with_suffix(".dropped.csv")

    speed_ratio = len(selected_names) / max(1, len(names))
    estimated_epoch_minutes = args.reference_epoch_minutes * speed_ratio
    estimated_total_hours = estimated_epoch_minutes * args.planned_epochs / 60.0
    report = {
        "processed_root": str(processed_root),
        "source_list": str(source_list),
        "output_list": str(output_list),
        "source_count": len(names),
        "filtered_count": len(rows),
        "selected_count": len(selected_names),
        "dropped_count": len(dropped),
        "seed": args.seed,
        "filters": {
            "min_corners": args.min_corners,
            "max_corners": args.max_corners,
            "max_edges": args.max_edges,
        },
        "bins": {key: len(value) for key, value in bins.items()},
        "target_by_bin": target_by_bin,
        "all_filtered_summary": summarize(rows),
        "selected_summary": summarize(selected_rows),
        "speed_estimate": {
            "reference_epoch_minutes": args.reference_epoch_minutes,
            "planned_epochs": args.planned_epochs,
            "estimated_epoch_minutes": estimated_epoch_minutes,
            "estimated_total_hours": estimated_total_hours,
        },
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    if args.write_stats_csv:
        with open(stats_csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["name"])
            writer.writeheader()
            writer.writerows(rows)

    if dropped:
        fieldnames = sorted({key for row in dropped for key in row.keys()})
        with open(dropped_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(dropped)

    print("")
    print("Done")
    print("source samples: {}".format(len(names)))
    print("filtered samples: {}".format(len(rows)))
    print("selected samples: {}".format(len(selected_names)))
    print("dropped samples: {}".format(len(dropped)))
    print("output_list: {}".format(output_list))
    print("report: {}".format(report_path))
    print("estimated epoch minutes: {:.2f}".format(estimated_epoch_minutes))
    print("estimated total hours for {} epochs: {:.2f}".format(args.planned_epochs, estimated_total_hours))


if __name__ == "__main__":
    main()
