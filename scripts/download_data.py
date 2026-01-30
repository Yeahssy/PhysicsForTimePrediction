#!/usr/bin/env python
"""Download benchmark datasets."""

import argparse

from src.data.download import (
    download_dataset,
    download_all_datasets,
    list_available_datasets,
)


def main():
    parser = argparse.ArgumentParser(description="Download time series datasets")

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset to download (or 'all' for all datasets)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory to save datasets",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file exists",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets",
    )

    args = parser.parse_args()

    if args.list:
        print("Available datasets:")
        for name in list_available_datasets():
            print(f"  - {name}")
        return

    if args.dataset is None or args.dataset == "all":
        print("Downloading all datasets...")
        download_all_datasets(args.data_dir, args.force)
    else:
        print(f"Downloading {args.dataset}...")
        download_dataset(args.dataset, args.data_dir, args.force)

    print("Done!")


if __name__ == "__main__":
    main()
