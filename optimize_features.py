#!/usr/bin/env python3
"""
Feature Optimization Script
Removes zero-importance features and consolidates redundant ones
"""

import pandas as pd
import json
from pathlib import Path

# Paths
MODELS_DIR = Path("models/trained_models")
SPREAD_IMPORTANCE_CSV = MODELS_DIR / "spread_feature_importance.csv"
ML_IMPORTANCE_CSV = MODELS_DIR / "ml_feature_importance.csv"
FEATURE_COLUMNS_JSON = MODELS_DIR / "true_hybrid_feature_columns.json"

def analyze_features():
    """Analyze feature importance and identify candidates for removal"""
    print("=" * 80)
    print("🔍 FEATURE OPTIMIZATION ANALYSIS")
    print("=" * 80)
    
    # Load spread feature importance
    spread_df = pd.read_csv(SPREAD_IMPORTANCE_CSV)
    
    # Identify zero-importance features
    zero_importance = spread_df[spread_df['importance'] == 0.0]['feature'].tolist()
    
    print(f"\n📊 Total Features: {len(spread_df)}")
    print(f"❌ Zero-importance features: {len(zero_importance)}")
    print(f"✅ Useful features: {len(spread_df) - len(zero_importance)}")
    
    # Show zero-importance features by category
    zero_df = spread_df[spread_df['importance'] == 0.0]
    print("\n🗑️  ZERO-IMPORTANCE FEATURES BY CATEGORY:")
    for category in zero_df['category'].unique():
        features = zero_df[zero_df['category'] == category]['feature'].tolist()
        print(f"\n  {category} ({len(features)} features):")
        for feat in features:
            print(f"    - {feat}")
    
    # Identify redundant feature groups
    print("\n\n🔄 REDUNDANT FEATURE GROUPS:")
    
    redundant_groups = {
        "Market Odds": [
            "market_home_ml_prob",
            "market_away_ml_prob", 
            "market_prob_diff"  # Keep this, drop others
        ],
        "Record Stats": [
            "home_wins",
            "away_wins",
            "home_win_pct",
            "away_win_pct",
            "record_diff",  # Keep this
            "home_losses",
            "away_losses"
        ],
        "Injury Impact": [
            "home_total_injury_impact",
            "away_total_injury_impact",
            "combined_injury_impact",
            "injury_impact_differential"  # Keep this
        ]
    }
    
    for group_name, features in redundant_groups.items():
        print(f"\n  {group_name}:")
        for feat in features:
            if feat in spread_df['feature'].values:
                importance = spread_df[spread_df['feature'] == feat]['importance'].values[0]
                pct = spread_df[spread_df['feature'] == feat]['importance_pct'].values[0]
                print(f"    {'✓ KEEP' if any(k in feat for k in ['_diff', 'differential']) else '  drop'}: {feat:<40} {pct:>5}%")
    
    return zero_importance, redundant_groups

def generate_optimized_feature_list():
    """Generate optimized feature list"""
    print("\n\n" + "=" * 80)
    print("🎯 GENERATING OPTIMIZED FEATURE LIST")
    print("=" * 80)
    
    # Load current features
    spread_df = pd.read_csv(SPREAD_IMPORTANCE_CSV)
    
    # Features to remove
    zero_importance = spread_df[spread_df['importance'] == 0.0]['feature'].tolist()
    
    # Redundant features to drop (keep the differential/diff versions)
    redundant_to_drop = [
        'market_home_ml_prob',  # Keep market_prob_diff
        'market_away_ml_prob',  # Keep market_prob_diff
        'home_wins',  # Keep record_diff
        'away_wins',  # Keep record_diff
        'home_losses',  # Keep record_diff
        'away_losses',  # Keep record_diff
        'home_total_injury_impact',  # Keep injury_impact_differential
        'away_total_injury_impact',  # Keep injury_impact_differential
        'combined_injury_impact'  # Keep injury_impact_differential
    ]
    
    # Only drop redundant features if they actually exist and have low importance
    redundant_to_drop = [
        f for f in redundant_to_drop 
        if f in spread_df['feature'].values and 
        spread_df[spread_df['feature'] == f]['importance_pct'].values[0] < 2.0
    ]
    
    all_features_to_drop = list(set(zero_importance + redundant_to_drop))
    
    # Create optimized feature list
    optimized_features = spread_df[~spread_df['feature'].isin(all_features_to_drop)]['feature'].tolist()
    
    print(f"\n📊 Optimization Summary:")
    print(f"   Original features: {len(spread_df)}")
    print(f"   Zero-importance dropped: {len(zero_importance)}")
    print(f"   Redundant features dropped: {len(redundant_to_drop)}")
    print(f"   Total dropped: {len(all_features_to_drop)}")
    print(f"   ✅ Optimized features: {len(optimized_features)}")
    print(f"   📉 Reduction: {len(all_features_to_drop) / len(spread_df) * 100:.1f}%")
    
    # Save optimized feature list
    output_file = MODELS_DIR / "optimized_feature_list.json"
    with open(output_file, 'w') as f:
        json.dump({
            'features': optimized_features,
            'original_count': len(spread_df),
            'optimized_count': len(optimized_features),
            'dropped_count': len(all_features_to_drop),
            'dropped_features': {
                'zero_importance': zero_importance,
                'redundant': redundant_to_drop
            }
        }, f, indent=2)
    
    print(f"\n💾 Saved to: {output_file}")
    
    # Show top features after optimization
    print(f"\n🔥 TOP 20 FEATURES (AFTER OPTIMIZATION):")
    optimized_df = spread_df[spread_df['feature'].isin(optimized_features)].head(20)
    for i, (_, row) in enumerate(optimized_df.iterrows(), 1):
        print(f"   {i:2d}. {row['feature']:<40} {row['importance_pct']:>5}%  [{row['category']}]")
    
    return optimized_features, all_features_to_drop

def main():
    """Main execution"""
    print("\n🏈 NFL Model Feature Optimization\n")
    
    # Step 1: Analyze current features
    zero_features, redundant_groups = analyze_features()
    
    # Step 2: Generate optimized list
    optimized_features, dropped_features = generate_optimized_feature_list()
    
    print("\n\n" + "=" * 80)
    print("✅ OPTIMIZATION COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review optimized_feature_list.json")
    print("2. Retrain model with optimized features")
    print("3. Compare accuracy: should be same or better")
    print("\nTo retrain:")
    print("  python -m src.true_hybrid_model --optimize")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()



