#!/usr/bin/env bash
set -euo pipefail

check_tool() {
  local tool=$1
  if command -v "$tool" >/dev/null 2>&1; then
    printf '%-20s %s\n' "$tool" "$(command -v "$tool")"
  else
    printf '%-20s MISSING\n' "$tool" >&2
    return 1
  fi
}

printf 'MLIR metamorphic-testing environment smoke test\n'
check_tool cmake
check_tool clang
check_tool ninja

test -f /workspace/build/CMakeCache.txt
printf '%-20s %s\n' "CMake superbuild" "configured"

if [[ "${ENABLE_FUZZING:-ON}" == "ON" ]]; then
  check_tool afl-fuzz
  check_tool flex
  check_tool bison
  afl-fuzz -h >/dev/null 2>&1 || true
  flex --version >/dev/null
  bison --version >/dev/null
fi

if [[ "${BUILD_CONQUER_OPT:-ON}" == "ON" ]]; then
  check_tool conquer-opt
  conquer-opt --help >/dev/null
else
  printf '%-20s %s\n' "conquer-opt" "not requested"
fi

printf 'Smoke test passed.\n'
