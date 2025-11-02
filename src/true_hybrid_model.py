"""
True Hybrid Model - Uses Advanced Features Only (2015+)
Combines PBP/NGS features + Injury Features + Market Odds
Only uses data where we have full Next Gen Stats coverage
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

# Import config
try:
    from .config import XGBOOST_ADVANCED_PARAMS
except ImportError:
    from config import XGBOOST_ADVANCED_PARAMS

def print_section_header(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*80}")
    print(f"🏈 {title}")
    print(f"{'='*80}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrueHybridModel:
    """True hybrid model using only advanced features (2015+) with NGS coverage"""
    
    def __init__(self, 
                 advanced_data_path: str = "data/enhanced_features.csv"):
        self.advanced_data_path = advanced_data_path
        self.advanced_df = None
        self.hybrid_df = None
        
        # Model storage
        self.models = {}
        self.feature_columns = {}
        self.metadata = {}
        
    def load_data(self):
        """Load advanced dataset only (2015+ with NGS coverage)"""
        print_section_header("Loading Advanced Dataset (2015+ with NGS Coverage)")
        
        # Load advanced data (2015-2025) - only data with full feature coverage
        print("📊 Loading Advanced Data (PBP/NGS + Injuries + Market Odds)...")
        self.advanced_df = pd.read_csv(self.advanced_data_path)
        print(f"✅ Advanced: {len(self.advanced_df)} games")
        print(f"✅ Advanced seasons: {sorted(self.advanced_df['season'].unique())}")
        print(f"✅ Includes market odds features: {'market_home_ml_prob' in self.advanced_df.columns}")
        
    def create_true_hybrid_dataset(self):
        """Create hybrid dataset using only advanced features (2015+ with NGS coverage)"""
        print_section_header("Creating Hybrid Dataset (2015+ Only)")
        
        # Use advanced data directly - it already has all features including market odds
        hybrid_df = self.advanced_df.copy()
        
        # Create targets if not present
        if 'home_team_wins' not in hybrid_df.columns:
            hybrid_df['home_team_wins'] = (hybrid_df['home_score'] > hybrid_df['away_score']).astype(int)
        if 'ats_home_covers' not in hybrid_df.columns:
            hybrid_df['ats_home_covers'] = (
                (hybrid_df['home_score'] - hybrid_df['away_score']) > -hybrid_df['spread']
            ).astype(int)
        
        # Standardize team columns
        if 'team_home' not in hybrid_df.columns:
            hybrid_df['team_home'] = hybrid_df['home_team']
        if 'team_away' not in hybrid_df.columns:
            hybrid_df['team_away'] = hybrid_df['away_team']
        
        print(f"✅ Hybrid dataset created: {len(hybrid_df)} games (2015-2024)")
        
        # Select features - exclude metadata columns
        # Identify injury features
        injury_feature_keywords = [
            'injury', 'qb_out', 'top_wr_out', 'top_cb_out', 'lt_out', 
            'edge_out', 'ol_out', 'players_out', 'players_doubtful', 
            'players_questionable', 'premium_injuries'
        ]
        
        # Select all features except metadata
        available_features = [col for col in hybrid_df.columns 
                           if col not in ['season', 'team_home', 'team_away', 'spread', 
                                        'home_team_wins', 'ats_home_covers',
                                        'home_score', 'away_score', 'home_team', 'away_team',
                                        'game_id']]
        
        # Count features by category
        injury_features_count = len([f for f in available_features 
                                     if any(kw in f for kw in injury_feature_keywords)])
        market_features_count = len([f for f in available_features if 'market' in f])
        record_features_count = len([f for f in available_features if 'win' in f or 'record' in f])
        
        print(f"📊 Total hybrid features: {len(available_features)}")
        print(f"📊 Market odds features: {market_features_count}")
        print(f"📊 Record features: {record_features_count}")
        print(f"📊 Injury features: {injury_features_count}")
        
        # Store the hybrid dataset
        self.hybrid_df = hybrid_df
        self.hybrid_features = available_features
        
        # Show feature breakdown
        print(f"\n🔧 Feature Breakdown:")
        print(f"   Market Odds: {market_features_count} features")
        print(f"   Record Features: {record_features_count} features")
        print(f"   PBP Features: {len([f for f in available_features if 'epa' in f or 'success' in f])} features")
        print(f"   NGS Features: {len([f for f in available_features if 'ngs' in f or 'time_to_throw' in f])} features")
        print(f"   Team Stats: {len([f for f in available_features if 'passing' in f or 'rushing' in f])} features")
        print(f"   Injury Features: {injury_features_count} features")
        print(f"   Rest Days: {len([f for f in available_features if 'rest' in f])} features")
        
    def train_hybrid_models(self):
        """Train true hybrid models with combined features"""
        print_section_header("Training True Hybrid Models")
        
        df = self.hybrid_df.copy()
        
        # Prepare features
        X = df[self.hybrid_features].select_dtypes(include=[np.number])
        y_ml = df['home_team_wins']  # Use target from advanced data
        y_spread = df['ats_home_covers']  # Use target from advanced data
        
        print(f"📊 Feature matrix: {X.shape}")
        print(f"📊 ML target distribution: {y_ml.value_counts().to_dict()}")
        print(f"📊 Spread target distribution: {y_spread.value_counts().to_dict()}")
        
        # Temporal split: Train on 2015-2022, Test on 2023-2024
        train_mask = df['season'] <= 2022
        test_mask = df['season'] >= 2023
        
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_ml_train = y_ml[train_mask]
        y_ml_test = y_ml[test_mask]
        y_spread_train = y_spread[train_mask]
        y_spread_test = y_spread[test_mask]
        
        print(f"✅ Training: {len(X_train)} games (2015-2022)")
        print(f"✅ Testing: {len(X_test)} games (2023-2024)")
        
        # Train ML model
        print("\n🤖 Training Hybrid ML Model...")
        ml_model = xgb.XGBClassifier(**XGBOOST_ADVANCED_PARAMS)
        ml_model.fit(X_train, y_ml_train)
        ml_pred = ml_model.predict(X_test)
        ml_acc = accuracy_score(y_ml_test, ml_pred)
        
        # Train Spread model
        print("🤖 Training Hybrid Spread Model...")
        spread_model = xgb.XGBClassifier(**XGBOOST_ADVANCED_PARAMS)
        spread_model.fit(X_train, y_spread_train)
        spread_pred = spread_model.predict(X_test)
        spread_acc = accuracy_score(y_spread_test, spread_pred)
        
        # Store models
        self.models['hybrid_ml'] = ml_model
        self.models['hybrid_spread'] = spread_model
        self.feature_columns['hybrid'] = list(X.columns)
        
        print(f"\n🎯 HYBRID MODEL PERFORMANCE:")
        print(f"   ML Accuracy: {ml_acc:.3f}")
        print(f"   Spread Accuracy: {spread_acc:.3f}")
        
        # Feature importance
        print(f"\n🔥 TOP 20 MOST IMPORTANT FEATURES:")
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': ml_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for i, (_, row) in enumerate(feature_importance.head(20).iterrows(), 1):
            # Highlight injury features
            is_injury = any(kw in row['feature'] for kw in ['injury', 'qb_out', 'wr_out', 'cb_out', 'lt_out', 'edge_out'])
            marker = "🏥" if is_injury else "  "
            print(f"   {marker} {i:2d}. {row['feature']:<35} {row['importance']:.4f}")
        
        # Show injury feature importance summary
        injury_keywords = ['injury', 'qb_out', 'wr_out', 'cb_out', 'lt_out', 'edge_out', 'ol_out']
        injury_features = feature_importance[
            feature_importance['feature'].str.contains('|'.join(injury_keywords), case=False, regex=True)
        ]
        
        if len(injury_features) > 0:
            print(f"\n🏥 INJURY FEATURE IMPORTANCE:")
            print(f"   Total injury features in top 50: {len(injury_features.head(50))}")
            print(f"   Average injury feature importance: {injury_features['importance'].mean():.4f}")
            print(f"   Top injury features:")
            for i, (_, row) in enumerate(injury_features.head(5).iterrows(), 1):
                print(f"      {i}. {row['feature']:<35} {row['importance']:.4f}")
        
    def save_models(self):
        """Save true hybrid models"""
        print_section_header("Saving True Hybrid Models")
        
        models_dir = Path("models/trained_models")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            model_path = models_dir / f"true_hybrid_{name}_model.joblib"
            joblib.dump(model, model_path)
            logger.info(f"✅ {name} model saved: {model_path}")
        
        # Save feature columns
        features_path = models_dir / "true_hybrid_feature_columns.json"
        with open(features_path, 'w') as f:
            json.dump(self.feature_columns, f, indent=2)
        logger.info(f"✅ Feature columns saved: {features_path}")
        
        # Save metadata
        injury_keywords = ['injury', 'qb_out', 'wr_out', 'cb_out', 'lt_out', 'edge_out', 'ol_out']
        
        metadata = {
            'hybrid_games': len(self.hybrid_df),
            'hybrid_features': len(self.hybrid_features),
            'feature_breakdown': {
                'market_odds': len([f for f in self.hybrid_features if 'market' in f]),
                'record_features': len([f for f in self.hybrid_features if 'win' in f or 'record' in f]),
                'pbp_features': len([f for f in self.hybrid_features if 'epa' in f or 'success' in f]),
                'ngs_features': len([f for f in self.hybrid_features if 'ngs' in f or 'time_to_throw' in f]),
                'team_stats': len([f for f in self.hybrid_features if 'passing' in f or 'rushing' in f]),
                'injury_features': len([f for f in self.hybrid_features if any(kw in f for kw in injury_keywords)]),
                'rest_days': len([f for f in self.hybrid_features if 'rest' in f])
            },
            'seasons': [int(x) for x in sorted(self.hybrid_df['season'].unique())],
            'models_trained': list(self.models.keys()),
            'includes_injuries': any(any(kw in f for kw in injury_keywords) for f in self.hybrid_features),
            'data_coverage': '2015+ (NGS coverage only)'
        }
        
        metadata_path = models_dir / "true_hybrid_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✅ Metadata saved: {metadata_path}")
        
    def run_full_training(self):
        """Run complete true hybrid model training"""
        print_section_header("🏈 TRUE HYBRID MODEL TRAINING")
        
        self.load_data()
        self.create_true_hybrid_dataset()
        self.train_hybrid_models()
        self.save_models()
        
        print_section_header("🎉 TRUE HYBRID MODEL COMPLETE!")
        print("✅ Model uses PBP/NGS features + Injury data + Market Odds (2015+)")
        print("✅ Only uses data with full Next Gen Stats coverage")
        print("✅ Injury-aware predictions with market-anchored probabilities")


if __name__ == "__main__":
    trainer = TrueHybridModel()
    trainer.run_full_training()
