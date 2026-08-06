#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = Path(os.environ.get("MLIR_MR_BUILD_DIR", PROJECT_ROOT / "build"))
BINARY_NAME = "mlir_mr_opt"

CANDIDATE_PATHS = [
    BUILD_DIR / "mlir-mr" / "src" / "app" / BINARY_NAME,
    BUILD_DIR / "bin" / BINARY_NAME,
    BUILD_DIR / BINARY_NAME,
    BUILD_DIR / "mlir-mr" / BINARY_NAME,
]


def find_binary():
    override = os.environ.get("MLIR_MR_OPT")
    if override:
        return Path(override)
    for path in CANDIDATE_PATHS:
        if path.is_file() and os.access(path, os.X_OK):
            return path
    return None


def build_binary():
    cmake = shutil.which("cmake")
    if not cmake:
        sys.exit("cmake not found in PATH; cannot build mlir_mr_opt")
    if not (BUILD_DIR / "CMakeCache.txt").exists():
        subprocess.run(
            [cmake, "-G", "Ninja", "-S", str(PROJECT_ROOT), "-B", str(BUILD_DIR)],
            check=True,
        )
    subprocess.run(
        [cmake, "--build", str(BUILD_DIR), "--target", BINARY_NAME, "-j"],
        check=True,
    )
    binary = find_binary()
    if not binary:
        sys.exit(f"build succeeded but {BINARY_NAME} was not found under {BUILD_DIR}")
    return binary


def collect_mlir_files(folder: Path):
    """Return list of .mlir files in the given folder (non-recursive)."""
    if not folder.is_dir():
        sys.exit(f"folder not found or not a directory: {folder}")
    files = [f for f in folder.iterdir() if f.is_file() and f.suffix == ".mlir"]
    if not files:
        sys.exit(f"no .mlir files found in {folder}")
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Run the mlir-mr-opt metamorphic pass pipeline on a single .mlir file or a corpus folder."
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="path to a single .mlir input file (ignored if --folder is used)",
    )
    parser.add_argument(
        "--folder",
        help="path to a folder of .mlir files; one is selected at random for each run",
    )
    parser.add_argument(
        "--transforms", default="load-reordering,insert-fence"
    )
    parser.add_argument("--runs", type=int, default=1, help="number of runs to perform")
    parser.add_argument("--print-mlir", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--binary", help="path to the mlir_mr_opt binary (overrides lookup)"
    )
    parser.add_argument(
        "--once",
        nargs="?",
        const=1,
        type=int,
        help="Run once with the given seed (default seed 1)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Prints all output from the binary"
    )

    args = parser.parse_args()

    if args.folder and args.input:
        sys.exit("Cannot specify both --folder and a positional input file.")
    if not args.folder and not args.input:
        sys.exit("You must provide either an input file or --folder.")

    if args.folder:
        folder_path = Path(args.folder)
        mlir_files = collect_mlir_files(folder_path)

        # function def: return a random file from the corpus each time called
        def get_input():
            return random.choice(mlir_files)
    else:
        single_file = Path(args.input)
        if not single_file.is_file():
            sys.exit(f"input file not found: {args.input}")
        def get_input():
            return single_file

    # Binary setup
    binary = Path(args.binary) if args.binary else find_binary()
    if not binary:
        print(f"{BINARY_NAME} not found; building it...", file=sys.stderr)
        binary = build_binary()

    runs = args.runs if args.once is None else 1

    for i in range(runs):
        # Get the input file either from the folder or the single file
        input_file = get_input()
        seed = random.randint(0, 2**32 - 1) if args.once is None else args.once

        cmd = [
            str(binary),
            f"--seed={seed}",
            f"--transforms={args.transforms}",
        ]
        if args.print_mlir:
            cmd.append("--print-mlir")
        if args.debug:
            cmd.append("--debug")
        cmd.append(str(input_file))

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(
                f"[CRASH] Run {i+1}, seed {seed}, file {input_file.name}, exit code {result.returncode}"
            )
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            if args.verbose and result.stdout:
                print(result.stdout)

        elif args.verbose:
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())