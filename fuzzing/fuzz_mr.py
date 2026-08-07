#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
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
        sys.exit("cmake not found in PATH")
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
        sys.exit(f"build succeeded but {BINARY_NAME} not found")
    return binary


def main():
    parser = argparse.ArgumentParser(
        description="Run the mlir-mr-opt metamorphic pass pipeline."
    )
    parser.add_argument(
        "file", nargs="?", help="path to a single .mlir file"
    )
    parser.add_argument(
        "--multi", metavar="FOLDER",
        help="folder of .mlir files; random one selected per run"
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--print-mlir", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--binary", help="override path to mlir_mr_opt")
    parser.add_argument("--seed", type=int, help="fixed seed for all runs")
    parser.add_argument("--transform", help="specific transformation to apply")
    parser.add_argument(
        "--output-dir", help="write MLIR output to directory"
    )

    args = parser.parse_args()

    if args.file and args.multi:
        sys.exit("Cannot specify both FILE and --multi.")
    if not args.file and not args.multi:
        sys.exit("Must provide either FILE or --multi FOLDER.")
    if args.output_dir and args.multi:
        sys.exit("Cannot use --output-dir with --multi.")
    if args.print_mlir and args.runs > 1:
        sys.exit("--print-mlir requires exactly one run.")

    binary = Path(args.binary) if args.binary else find_binary()
    if not binary:
        print(f"{BINARY_NAME} not found; building...", file=sys.stderr)
        binary = build_binary()

    cmd = [str(binary)]
    if args.print_mlir:
        cmd.append("--print-mlir")
    if args.seed is not None:
        cmd.append(f"--seed={args.seed}")
    if args.transform:
        cmd.append(f"--transform={args.transform}")
    cmd.append(f"--runs={args.runs}")
    if args.multi:
        cmd.append(f"--multi={args.multi}")
    if args.file:
        cmd.append(args.file)

    result = subprocess.run(cmd, capture_output=True, text=True)

    # just print non run-info errors (ones that i cant catch)
    if result.stderr:
        sys.stderr.write(result.stderr)

    runs = []
    raw = result.stdout
    start = raw.find("[")
    end = raw.rfind("]")

    # parse the JSON array of run info from stdout
    if start != -1 and end != -1 and end > start:
        try:
            runs = json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            sys.stderr.write("failed to parse JSON from stdout\n")
            sys.stderr.write(raw)
            sys.exit(1)
    elif raw.strip():
        sys.stderr.write("no JSON array found in stdout\n")
        sys.stderr.write(raw)
        sys.exit(1)

    # print run info
    for run in runs:
        is_error = run.get("error", "") != ""
        if args.verbose or is_error:
            status = "[FAIL]" if is_error else "[OK]"
            xform = run.get("applied_transform", "none")
            print(f"{status} run {run['run']}, seed {run['seed']}, "
                  f"file {run['file']}, transform: {xform}")
            if is_error:
                print(f"       error: {run['error']}")

    # if singular run and --print-mlir, print the MLIR output to stdout
    if args.print_mlir and runs:
        mlir = runs[0].get("mlir_output", "")
        if mlir:
            sys.stdout.write(mlir)

    # if singular run and --output-dir, write the MLIR output to a file in the specified directory
    if args.output_dir and runs:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for run in runs:
            mlir = run.get("mlir_output", "")
            if mlir:
                stem = Path(run["file"]).stem
                out_file = (
                    out_dir / f"{stem}_run{run['run']}_seed{run['seed']}.mlir"
                )
                out_file.write_text(mlir)


if __name__ == "__main__":
    sys.exit(main())