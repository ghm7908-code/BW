import argparse
from pathlib import Path

from proj import process_one_sample, resolve_sample_names


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a unified BWFormer training root from Tallinn raw train/val splits. "
            "The output root will contain rgb/, annot/, vis/, train_list.txt, valid_list.txt, and all_list.txt."
        )
    )
    parser.add_argument(
        "--raw_root",
        required=True,
        help="Raw Tallinn root containing split directories such as train/ and val/.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Unified processed BWFormer dataset root to create.",
    )
    parser.add_argument("--train_split", default="train", help="Training split directory name under raw_root.")
    parser.add_argument("--val_split", default="val", help="Validation split directory name under raw_root.")
    parser.add_argument("--pc_subdir", default="xyz", help="Point cloud subdirectory name under each split.")
    parser.add_argument(
        "--wireframe_subdir",
        default="gt",
        help="Preferred wireframe subdirectory name under each split.",
    )
    parser.add_argument(
        "--fallback_wireframe_subdir",
        default="wireframe",
        help="Fallback wireframe subdirectory name if --wireframe_subdir does not exist.",
    )
    parser.add_argument("--image_size", default=256, type=int, help="Projected BWFormer image size.")
    parser.add_argument(
        "--projection_mode",
        default="standard",
        choices=["standard", "roof_prior_v1", "roof_prior_v1b"],
        help="Projection strategy forwarded to proj.py.",
    )
    parser.add_argument("--top_band_px", default=8.0, type=float)
    parser.add_argument("--blur_kernel", default=5, type=int)
    parser.add_argument("--height_gamma", default=0.55, type=float)
    parser.add_argument("--valid_floor", default=0.18, type=float)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite projected rgb/annot/vis outputs if they already exist.",
    )
    return parser.parse_args()


def resolve_wireframe_dir(split_root, preferred_name, fallback_name):
    preferred = split_root / preferred_name
    if preferred.exists():
        return preferred

    fallback = split_root / fallback_name
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        "Could not find wireframe directory under {}. Tried '{}' and '{}'.".format(
            split_root,
            preferred_name,
            fallback_name,
        )
    )


def ensure_no_collisions(train_names, val_names):
    overlap = set(train_names) & set(val_names)
    if overlap:
        raise RuntimeError(
            "Found {} duplicate sample ids across train/val, e.g. {}".format(
                len(overlap),
                sorted(list(overlap))[:10],
            )
        )


def write_list(path, names):
    with open(path, "w", encoding="utf-8") as handle:
        for name in names:
            handle.write(name + "\n")


def project_split(
    split_name,
    split_root,
    pc_subdir,
    wireframe_dir,
    output_root,
    image_size,
    projection_mode,
    top_band_px,
    blur_kernel,
    height_gamma,
    valid_floor,
    overwrite,
):
    pc_dir = split_root / pc_subdir
    if not pc_dir.exists():
        raise FileNotFoundError("Point cloud directory does not exist: {}".format(pc_dir))

    sample_names, missing_pc, missing_wireframe = resolve_sample_names(pc_dir, wireframe_dir, names_file="")
    if not sample_names:
        raise RuntimeError("No matched .xyz/.obj pairs found for split '{}' under {}".format(split_name, split_root))

    rgb_dir = output_root / "rgb"
    annot_dir = output_root / "annot"
    vis_dir = output_root / "vis"

    rgb_dir.mkdir(parents=True, exist_ok=True)
    annot_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for index, name in enumerate(sample_names, start=1):
        try:
            process_one_sample(
                name=name,
                pc_dir=pc_dir,
                wireframe_dir=wireframe_dir,
                rgb_dir=rgb_dir,
                annot_dir=annot_dir,
                vis_dir=vis_dir,
                image_size=image_size,
                projection_mode=projection_mode,
                top_band_px=top_band_px,
                blur_kernel=blur_kernel,
                height_gamma=height_gamma,
                valid_floor=valid_floor,
                overwrite=overwrite,
            )
        except Exception as exc:
            failed.append((name, str(exc)))

        if index % 100 == 0 or index == len(sample_names):
            print("[{}] processed {}/{} samples".format(split_name, index, len(sample_names)))

    print("[{}] matched samples: {}".format(split_name, len(sample_names)))
    print("[{}] missing point clouds: {}".format(split_name, len(missing_pc)))
    print("[{}] missing wireframes: {}".format(split_name, len(missing_wireframe)))
    print("[{}] failed samples: {}".format(split_name, len(failed)))

    if failed:
        failed_log = output_root / "{}_failed_samples.txt".format(split_name)
        with open(failed_log, "w", encoding="utf-8") as handle:
            for name, reason in failed:
                handle.write("{}\t{}\n".format(name, reason))
        print("[{}] wrote failure log: {}".format(split_name, failed_log))

    successful_names = [name for name in sample_names if name not in {item[0] for item in failed}]
    return successful_names


def main():
    args = parse_args()

    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    train_root = raw_root / args.train_split
    val_root = raw_root / args.val_split

    if not raw_root.exists():
        raise FileNotFoundError("Raw root does not exist: {}".format(raw_root))
    if not train_root.exists():
        raise FileNotFoundError("Train split does not exist: {}".format(train_root))
    if not val_root.exists():
        raise FileNotFoundError("Validation split does not exist: {}".format(val_root))

    output_root.mkdir(parents=True, exist_ok=True)

    train_wireframe_dir = resolve_wireframe_dir(train_root, args.wireframe_subdir, args.fallback_wireframe_subdir)
    val_wireframe_dir = resolve_wireframe_dir(val_root, args.wireframe_subdir, args.fallback_wireframe_subdir)

    train_names = project_split(
        split_name=args.train_split,
        split_root=train_root,
        pc_subdir=args.pc_subdir,
        wireframe_dir=train_wireframe_dir,
        output_root=output_root,
        image_size=args.image_size,
        projection_mode=args.projection_mode,
        top_band_px=args.top_band_px,
        blur_kernel=args.blur_kernel,
        height_gamma=args.height_gamma,
        valid_floor=args.valid_floor,
        overwrite=args.overwrite,
    )
    val_names = project_split(
        split_name=args.val_split,
        split_root=val_root,
        pc_subdir=args.pc_subdir,
        wireframe_dir=val_wireframe_dir,
        output_root=output_root,
        image_size=args.image_size,
        projection_mode=args.projection_mode,
        top_band_px=args.top_band_px,
        blur_kernel=args.blur_kernel,
        height_gamma=args.height_gamma,
        valid_floor=args.valid_floor,
        overwrite=args.overwrite,
    )

    ensure_no_collisions(train_names, val_names)

    all_names = sorted(train_names + val_names)
    train_list_path = output_root / "train_list.txt"
    valid_list_path = output_root / "valid_list.txt"
    all_list_path = output_root / "all_list.txt"

    write_list(train_list_path, sorted(train_names))
    write_list(valid_list_path, sorted(val_names))
    write_list(all_list_path, all_names)

    print("")
    print("Done")
    print("output_root: {}".format(output_root))
    print("train samples: {}".format(len(train_names)))
    print("valid samples: {}".format(len(val_names)))
    print("all samples: {}".format(len(all_names)))
    print("train_list: {}".format(train_list_path))
    print("valid_list: {}".format(valid_list_path))
    print("all_list: {}".format(all_list_path))


if __name__ == "__main__":
    main()
