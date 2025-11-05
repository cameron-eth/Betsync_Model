#!/bin/bash
# Quick script to run NFL results backfill
# Usage: ./run_backfill.sh [days_back]
# Example: ./run_backfill.sh 14  (backfill last 14 days)

DAYS_BACK=${1:-7}

echo "🏈 Running NFL Results Backfill"
echo "📅 Looking back: $DAYS_BACK days"
echo ""

python3 backfill_nfl_results.py $DAYS_BACK

