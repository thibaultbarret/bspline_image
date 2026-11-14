#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Usage: check_coverage.sh <lcov.info> <min-line-coverage-percent>
# Fails if the line coverage reported by `lcov --summary` is below the threshold.
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <lcov.info> <threshold-percent>" >&2
  exit 2
fi
info_file=$1
threshold=$2

summary=$(lcov --summary "$info_file" 2>&1)
echo "$summary"

percent=$(printf '%s\n' "$summary" \
  | sed -n 's/^[[:space:]]*lines\.*:[[:space:]]*\([0-9][0-9.]*\)%.*/\1/p' | head -n 1)
if [ -z "$percent" ]; then
  echo "error: could not parse line coverage from lcov summary" >&2
  exit 1
fi

echo "Line coverage: ${percent}% (threshold: ${threshold}%)"
if awk -v p="$percent" -v t="$threshold" 'BEGIN { exit !(p + 0 < t + 0) }'; then
  echo "error: line coverage ${percent}% is below the required ${threshold}%" >&2
  exit 1
fi
echo "Coverage check passed."
