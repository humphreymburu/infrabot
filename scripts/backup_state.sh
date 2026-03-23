#!/usr/bin/env bash
set -euo pipefail

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${1:-/tmp/tech_advisor_backups}"
mkdir -p "$OUT_DIR"

TRACE_DIR="${TRACE_DIR:-/tmp/tech_advisor_traces}"
LINEAGE_DIR="${LINEAGE_DIR:-/tmp/tech_advisor_lineage}"
REVIEW_DIR="${REVIEW_DIR:-/tmp/tech_advisor_reviews}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/tmp/tech_advisor_checkpoints}"

ARCHIVE="$OUT_DIR/tech_advisor_state_${TS}.tar.gz"
tar -czf "$ARCHIVE" \
  -C / \
  "${TRACE_DIR#/}" \
  "${LINEAGE_DIR#/}" \
  "${REVIEW_DIR#/}" \
  "${CHECKPOINT_DIR#/}" 2>/dev/null || true

echo "Backup created: $ARCHIVE"
