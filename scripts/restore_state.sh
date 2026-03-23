#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 /path/to/tech_advisor_state_YYYYMMDD_HHMMSS.tar.gz"
  exit 1
fi

ARCHIVE="$1"
if [ ! -f "$ARCHIVE" ]; then
  echo "Archive not found: $ARCHIVE"
  exit 1
fi

tar -xzf "$ARCHIVE" -C /
echo "Restore complete from: $ARCHIVE"
