#!/usr/bin/env python3
"""Scan a campaign directory for runs whose outcome counts fall in (LOW, HIGH)."""
import argparse
import json
import sys
from glob import glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_LOW = 32
DEFAULT_HIGH = 4950


def resolve(campaign_dir):
    path = Path(campaign_dir)
    return path if path.is_absolute() else PROJECT_ROOT / path


def outcome_hits(run_info, low, high):
    hits = []
    for tg in run_info.get("thread_results", []):
        threads = tg.get("threads")
        os_ = tg.get("outcome_set", {})
        for entry in os_.get("transformed_outcomes", []):
            count = entry.get("count")
            if count is not None and low < count < high:
                hits.append((threads, entry.get("outcome", []), count))
    return hits


def main():
    parser = argparse.ArgumentParser(
        description="Find runs in a campaign with any outcome count strictly "
                    "between --low and --high."
    )
    parser.add_argument("campaign_dir", help="campaign directory to scan")
    parser.add_argument("--low", type=int, default=DEFAULT_LOW,
                        help=f"exclusive lower bound (default {DEFAULT_LOW})")
    parser.add_argument("--high", type=int, default=DEFAULT_HIGH,
                        help=f"exclusive upper bound (default {DEFAULT_HIGH})")
    parser.add_argument("--quiet", action="store_true",
                        help="only print matching run paths, no per-outcome lines")
    args = parser.parse_args()

    if args.low >= args.high:
        sys.exit("--low must be < --high")

    root = resolve(args.campaign_dir)
    if not root.is_dir():
        sys.exit(f"campaign dir not found: {root}")

    files = sorted(glob(str(root / "**" / "run_info.json"), recursive=True))
    if not files:
        print(f"no run_info.json files found under {root}", file=sys.stderr)
        return 1

    matched = 0
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"skipping unreadable {f}: {exc}", file=sys.stderr)
            continue
        hits = outcome_hits(d, args.low, args.high)
        if not hits:
            continue
        matched += 1
        rel = Path(f).relative_to(root)
        print(f"run {d.get('run', '?'):>4}  {rel}  ({d.get('file', '?')})")
        if not args.quiet:
            for threads, values, count in hits:
                print(f"    threads={threads}  trn outcome={values} "
                      f"count={count}")

    print(f"=== {matched}/{len(files)} runs have outcome counts in "
          f"({args.low}, {args.high}) ===")
    return 0 if matched else 1


if __name__ == "__main__":
    sys.exit(main())
