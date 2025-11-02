#!/bin/bash
# Run NFL Predictions with Injury-Aware Model
# Writes to Supabase: nfl_model_predictions table

cd /Users/cameron/Documents/APP-BUILDS/betsync/betsync_model

export ODDS_API_KEY="cd667934dd4a58aea4086b4acf177a32"

echo "🏈 Running NFL Predictions with Injury-Aware Model"
echo "=================================================="
echo ""
echo "Model includes:"
echo "  ✅ 2,743 games (2015-2024)"
echo "  ✅ 99 features (EPA, injuries, W/L records)"
echo "  ✅ No data leakage"
echo "  ✅ Injury-aware (32 features)"
echo ""

python run_pipeline.py --predict

echo ""
echo "✅ Predictions written to Supabase: nfl_model_predictions"






