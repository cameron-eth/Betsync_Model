"""
Display feature importance from trained models
"""

import pandas as pd
import joblib
import json
from pathlib import Path
import numpy as np

def print_section_header(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*80}")
    print(f"🏈 {title}")
    print(f"{'='*80}")

def categorize_feature(feature: str) -> str:
    """Categorize a feature by its name"""
    if 'injury' in feature.lower() or 'qb_out' in feature or 'wr_out' in feature or 'cb_out' in feature or 'lt_out' in feature or 'edge_out' in feature:
        return 'Injury'
    elif 'market' in feature.lower():
        return 'Market Odds'
    elif 'record' in feature.lower() or 'win' in feature.lower() or 'loss' in feature.lower():
        return 'Record'
    elif 'epa' in feature.lower() or 'success' in feature.lower() or 'explosive' in feature.lower():
        return 'PBP (EPA)'
    elif 'ngs' in feature.lower() or 'time_to_throw' in feature.lower() or 'air_yards' in feature.lower() or 'aggressiveness' in feature.lower():
        return 'Next Gen Stats'
    elif 'passing' in feature.lower() or 'rushing' in feature.lower() or 'sacks' in feature.lower() or 'penalties' in feature.lower():
        return 'Team Stats'
    elif 'rest' in feature.lower():
        return 'Rest Days'
    elif 'week' in feature.lower():
        return 'Context'
    else:
        return 'Other'

def show_feature_importance():
    """Load trained models and display feature importance"""
    print_section_header("MODEL FEATURE IMPORTANCE")
    
    models_dir = Path("models/trained_models")
    
    # Load feature columns
    features_path = models_dir / "true_hybrid_feature_columns.json"
    with open(features_path, 'r') as f:
        feature_data = json.load(f)
    
    feature_names = feature_data['hybrid']
    
    # Load models
    ml_model_path = models_dir / "true_hybrid_hybrid_ml_model.joblib"
    spread_model_path = models_dir / "true_hybrid_hybrid_spread_model.joblib"
    
    print(f"📊 Loading models from {models_dir}")
    ml_model = joblib.load(ml_model_path)
    spread_model = joblib.load(spread_model_path)
    
    print(f"✅ Loaded {len(feature_names)} features")
    
    # Get feature importance from both models
    ml_importance = ml_model.feature_importances_
    spread_importance = spread_model.feature_importances_
    
    # Create DataFrames
    ml_df = pd.DataFrame({
        'feature': feature_names,
        'importance': ml_importance,
        'category': [categorize_feature(f) for f in feature_names]
    }).sort_values('importance', ascending=False)
    
    spread_df = pd.DataFrame({
        'feature': feature_names,
        'importance': spread_importance,
        'category': [categorize_feature(f) for f in feature_names]
    }).sort_values('importance', ascending=False)
    
    # Normalize importance to percentages
    ml_df['importance_pct'] = (ml_df['importance'] / ml_df['importance'].sum() * 100).round(2)
    spread_df['importance_pct'] = (spread_df['importance'] / spread_df['importance'].sum() * 100).round(2)
    
    # Display ML Model Feature Importance
    print_section_header("ML MODEL (Home Team Wins) - Feature Importance")
    print(f"\n{'Rank':<6} {'Feature':<45} {'Importance':<12} {'%':<8} {'Category':<15}")
    print("-" * 90)
    
    for i, (_, row) in enumerate(ml_df.iterrows(), 1):
        marker = "🏥" if row['category'] == 'Injury' else "  "
        print(f"{marker} {i:<4} {row['feature']:<45} {row['importance']:<12.6f} {row['importance_pct']:<8.2f} {row['category']:<15}")
    
    # Display Spread Model Feature Importance
    print_section_header("SPREAD MODEL (ATS Home Covers) - Feature Importance")
    print(f"\n{'Rank':<6} {'Feature':<45} {'Importance':<12} {'%':<8} {'Category':<15}")
    print("-" * 90)
    
    for i, (_, row) in enumerate(spread_df.iterrows(), 1):
        marker = "🏥" if row['category'] == 'Injury' else "  "
        print(f"{marker} {i:<4} {row['feature']:<45} {row['importance']:<12.6f} {row['importance_pct']:<8.2f} {row['category']:<15}")
    
    # Summary by category
    print_section_header("Feature Importance Summary by Category")
    
    print("\n📊 ML MODEL - Category Importance:")
    ml_category_summary = ml_df.groupby('category').agg({
        'importance': 'sum',
        'importance_pct': 'sum',
        'feature': 'count'
    }).rename(columns={'feature': 'count'}).sort_values('importance', ascending=False)
    ml_category_summary['avg_importance'] = (ml_category_summary['importance'] / ml_category_summary['count']).round(6)
    
    print(f"\n{'Category':<20} {'Total %':<12} {'Avg Importance':<18} {'Feature Count':<15}")
    print("-" * 70)
    for category, row in ml_category_summary.iterrows():
        print(f"{category:<20} {row['importance_pct']:<12.2f} {row['avg_importance']:<18.6f} {int(row['count']):<15}")
    
    print("\n📊 SPREAD MODEL - Category Importance:")
    spread_category_summary = spread_df.groupby('category').agg({
        'importance': 'sum',
        'importance_pct': 'sum',
        'feature': 'count'
    }).rename(columns={'feature': 'count'}).sort_values('importance', ascending=False)
    spread_category_summary['avg_importance'] = (spread_category_summary['importance'] / spread_category_summary['count']).round(6)
    
    print(f"\n{'Category':<20} {'Total %':<12} {'Avg Importance':<18} {'Feature Count':<15}")
    print("-" * 70)
    for category, row in spread_category_summary.iterrows():
        print(f"{category:<20} {row['importance_pct']:<12.2f} {row['avg_importance']:<18.6f} {int(row['count']):<15}")
    
    # Top features comparison
    print_section_header("Top 20 Features Comparison")
    print("\nML Model Top 20:")
    print(f"{'Rank':<6} {'Feature':<50} {'ML %':<10} {'Spread %':<10}")
    print("-" * 80)
    for i, (_, row) in enumerate(ml_df.head(20).iterrows(), 1):
        spread_pct = spread_df[spread_df['feature'] == row['feature']]['importance_pct'].values[0]
        print(f"{i:<6} {row['feature']:<50} {row['importance_pct']:<10.2f} {spread_pct:<10.2f}")
    
    # Save to CSV
    output_dir = Path("models/trained_models")
    ml_df.to_csv(output_dir / "ml_feature_importance.csv", index=False)
    spread_df.to_csv(output_dir / "spread_feature_importance.csv", index=False)
    print(f"\n✅ Feature importance saved to CSV files:")
    print(f"   - {output_dir / 'ml_feature_importance.csv'}")
    print(f"   - {output_dir / 'spread_feature_importance.csv'}")

if __name__ == "__main__":
    show_feature_importance()




