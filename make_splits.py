import argparse
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate deterministic train/valid/test split files for BWFormer datasets.'
    )
    parser.add_argument(
        '--dataset_root',
        required=True,
        help='dataset root that contains all_list.txt or generated subdirectories such as annot/',
    )
    parser.add_argument(
        '--source_file',
        default='',
        help='optional explicit source file listing sample ids, one per line',
    )
    parser.add_argument(
        '--source_subdir',
        default='annot',
        help='fallback subdirectory to scan when source_file is not provided and all_list.txt is missing',
    )
    parser.add_argument('--source_suffix', default='.npy', help='suffix used when scanning source_subdir')
    parser.add_argument('--train_ratio', default=0.9, type=float, help='train split ratio')
    parser.add_argument('--valid_ratio', default=0.05, type=float, help='validation split ratio')
    parser.add_argument('--test_ratio', default=0.05, type=float, help='test split ratio')
    parser.add_argument('--seed', default=3407, type=int, help='random seed for deterministic shuffling')
    parser.add_argument('--no_shuffle', action='store_true', help='keep lexical order instead of shuffling')
    return parser.parse_args()


def load_names(dataset_root, source_file, source_subdir, source_suffix):
    if source_file:
        source_path = Path(source_file)
    else:
        default_all_list = dataset_root / 'all_list.txt'
        source_path = default_all_list if default_all_list.exists() else None

    if source_path is not None:
        if not source_path.exists():
            raise FileNotFoundError('Source file does not exist: {}'.format(source_path))
        with open(source_path, 'r', encoding='utf-8') as handle:
            return [line.strip() for line in handle if line.strip()]

    scan_dir = dataset_root / source_subdir
    if not scan_dir.exists():
        raise FileNotFoundError(
            'Could not find all_list.txt and fallback directory does not exist: {}'.format(scan_dir)
        )

    return sorted(path.stem for path in scan_dir.glob('*{}'.format(source_suffix)))


def validate_ratios(train_ratio, valid_ratio, test_ratio):
    total = train_ratio + valid_ratio + test_ratio
    if total <= 0:
        raise ValueError('Split ratios must sum to a positive number.')
    return train_ratio / total, valid_ratio / total, test_ratio / total


def split_names(names, train_ratio, valid_ratio, test_ratio, seed, shuffle):
    names = list(names)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(names)

    total = len(names)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)

    train_names = names[:train_end]
    valid_names = names[train_end:valid_end]
    test_names = names[valid_end:]

    if total >= 3:
        if len(valid_names) == 0:
            valid_names = [train_names.pop()]
        if len(test_names) == 0:
            test_names = [train_names.pop()]

    return train_names, valid_names, test_names


def write_split_file(path, names):
    with open(path, 'w', encoding='utf-8') as handle:
        for name in names:
            handle.write(name + '\n')


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError('Dataset root does not exist: {}'.format(dataset_root))

    train_ratio, valid_ratio, test_ratio = validate_ratios(
        args.train_ratio,
        args.valid_ratio,
        args.test_ratio,
    )
    names = load_names(
        dataset_root,
        args.source_file,
        args.source_subdir,
        args.source_suffix,
    )

    if not names:
        raise RuntimeError('No sample ids were found under {}'.format(dataset_root))

    train_names, valid_names, test_names = split_names(
        names,
        train_ratio,
        valid_ratio,
        test_ratio,
        args.seed,
        shuffle=not args.no_shuffle,
    )

    train_path = dataset_root / 'train_list.txt'
    valid_path = dataset_root / 'valid_list.txt'
    test_path = dataset_root / 'test_list.txt'

    write_split_file(train_path, train_names)
    write_split_file(valid_path, valid_names)
    write_split_file(test_path, test_names)

    print('Total samples: {}'.format(len(names)))
    print('Train: {} -> {}'.format(len(train_names), train_path))
    print('Valid: {} -> {}'.format(len(valid_names), valid_path))
    print('Test: {} -> {}'.format(len(test_names), test_path))


if __name__ == '__main__':
    main()
