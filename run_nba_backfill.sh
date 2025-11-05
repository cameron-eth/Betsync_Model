#!/bin/bash
# Quick script to run NBA results backfill
# Usage: ./run_nba_backfill.sh [days_back]
# Example: ./run_nba_backfill.sh 7  (backfill last 7 days)

DAYS_BACK=${1:-3}

echo "🏀 Running NBA Results Backfill"
echo "📅 Looking back: $DAYS_BACK days"
echo ""

python3 backfill_nba_results.py $DAYS_BACK

