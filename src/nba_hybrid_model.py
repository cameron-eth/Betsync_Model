"""
NBA Hybrid Model - Uses Advanced Features
Combines team stats + advanced metrics + injury features + market odds
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
    from .nba_config import (
        XGBOOST_ADVANCED_PARAMS, NBA_ENHANCED_FEATURES_CLEAN_CSV,
        NBA_ML_MODEL_PATH, NBA_SPREAD_MODEL_PATH,
        NBA_FEATURE_COLUMNS_PATH, NBA_METADATA_PATH
    )
    from .nba_utils import print_section_header
except ImportError:
    from nba_config import (
        XGBOOST_ADVANCED_PARAMS, NBA_ENHANCED_FEATURES_CLEAN_CSV,
        NBA_ML_MODEL_PATH, NBA_SPREAD_MODEL_PATH,
        NBA_FEATURE_COLUMNS_PATH, NBA_METADATA_PATH
    )
    from nba_utils import print_section_header

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NBAHybridModel:
    """NBA hybrid model using advanced features"""
    
    def __init__(self, advanced_data_path: str = None):
        if advanced_data_path is None:
            advanced_data_path = NBA_ENHANCED_FEATURES_CLEAN_CSV
        self.advanced_data_path = advanced_data_path
        self.advanced_df = None
        self.hybrid_df = None
        
        # Model storage
        self.models = {}
        self.feature_columns = {}
        self.metadata = {}
        
    def load_data(self):
        """Load enhanced dataset"""
        print_section_header("Loading Enhanced NBA Dataset")
        
        print(f"📊 Loading Enhanced Data...")
        self.advanced_df = pd.read_csv(self.advanced_data_path)
        print(f"✅ Loaded {len(self.advanced_df)} games")
        print(f"✅ Seasons: {sorted(self.advanced_df['season'].unique())}")
        
    def create_hybrid_dataset(self):
        """Create hybrid dataset using advanced features"""
        print_section_header("Creating NBA Hybrid Dataset")
        
        # Use advanced data directly
        hybrid_df = self.advanced_df.copy()
        
        # Create targets if not present
        if 'home_team_wins' not in hybrid_df.columns:
            hybrid_df['home_team_wins'] = (hybrid_df['home_score'] > hybrid_df['away_score']).astype(int)
        
        if 'ats_home_covers' not in hybrid_df.columns:
            if 'spread' in hybrid_df.columns:
                hybrid_df['ats_home_covers'] = (
                    (hybrid_df['home_score'] - hybrid_df['away_score']) > -hybrid_df['spread']
                ).astype(int)
            else:
                hybrid_df['ats_home_covers'] = hybrid_df['home_team_wins']  # Default to ML
        
        # Standardize team columns
        if 'team_home' not in hybrid_df.columns:
            hybrid_df['team_home'] = hybrid_df['home_team']
        if 'team_away' not in hybrid_df.columns:
            hybrid_df['team_away'] = hybrid_df['away_team']
        
        print(f"✅ Hybrid dataset created: {len(hybrid_df)} games")
        
        # Select features - exclude metadata columns
        available_features = [col for col in hybrid_df.columns 
                           if col not in ['season', 'team_home', 'team_away', 'spread', 
                                        'home_team_wins', 'ats_home_covers',
                                        'home_score', 'away_score', 'home_team', 'away_team',
                                        'game_id', 'game_date']]
        
        # Count features by category
        injury_features_count = len([f for f in available_features if 'injury' in f.lower()])
        record_features_count = len([f for f in available_features if 'win' in f.lower() or 'record' in f.lower()])
        shooting_features_count = len([f for f in available_features if 'fg' in f.lower() or 'shot' in f.lower()])
        pace_features_count = len([f for f in available_features if 'pace' in f.lower() or 'poss' in f.lower()])
        
        print(f"📊 Total hybrid features: {len(available_features)}")
        print(f"📊 Record features: {record_features_count}")
        print(f"📊 Shooting features: {shooting_features_count}")
        print(f"📊 Pace features: {pace_features_count}")
        print(f"📊 Injury features: {injury_features_count}")
        
        # Store the hybrid dataset
        self.hybrid_df = hybrid_df
        self.hybrid_features = available_features
        
        print(f"\n🔧 Feature Breakdown:")
        print(f"   Record Features: {record_features_count} features")
        print(f"   Shooting Features: {shooting_features_count} features")
        print(f"   Pace/Efficiency: {pace_features_count} features")
        print(f"   Injury Features: {injury_features_count} features")
        print(f"   Rest Days: {len([f for f in available_features if 'rest' in f.lower()])} features")
        
    def train_hybrid_models(self):
        """Train hybrid models"""
        print_section_header("Training NBA Hybrid Models")
        
        df = self.hybrid_df.copy()
        
        # Prepare features
        X = df[self.hybrid_features].select_dtypes(include=[np.number])
        
        # Handle NaN values
        X = X.fillna(X.median())
        
        y_ml = df['home_team_wins']
        y_spread = df['ats_home_covers']
        
        print(f"📊 Feature matrix: {X.shape}")
        print(f"📊 ML target distribution: {y_ml.value_counts().to_dict()}")
        print(f"📊 Spread target distribution: {y_spread.value_counts().to_dict()}")
        
        # Temporal split: Train on older seasons, Test on recent seasons
        # Use 80/20 split by season
        seasons = sorted(df['season'].unique())
        
        if len(seasons) == 1:
            # Only one season - use train_test_split instead
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_ml_train, y_ml_test, y_spread_train, y_spread_test = train_test_split(
                X, y_ml, y_spread, test_size=0.2, random_state=42
            )
            print(f"✅ Training: {len(X_train)} games (80% split)")
            print(f"✅ Testing: {len(X_test)} games (20% split)")
        else:
            split_idx = int(len(seasons) * 0.8)
            train_seasons = seasons[:split_idx]
            test_seasons = seasons[split_idx:]
            
            train_mask = df['season'].isin(train_seasons)
            test_mask = df['season'].isin(test_seasons)
            
            X_train = X[train_mask]
            X_test = X[test_mask]
            y_ml_train = y_ml[train_mask]
            y_ml_test = y_ml[test_mask]
            y_spread_train = y_spread[train_mask]
            y_spread_test = y_spread[test_mask]
            
            print(f"✅ Training: {len(X_train)} games (seasons {min(train_seasons)}-{max(train_seasons)})")
            print(f"✅ Testing: {len(X_test)} games (seasons {min(test_seasons)}-{max(test_seasons)})")
        
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
        
        print(f"\n🎯 NBA HYBRID MODEL PERFORMANCE:")
        print(f"   ML Accuracy: {ml_acc:.3f}")
        print(f"   Spread Accuracy: {spread_acc:.3f}")
        
        # Feature importance
        print(f"\n🔥 TOP 20 MOST IMPORTANT FEATURES:")
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': ml_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for i, (_, row) in enumerate(feature_importance.head(20).iterrows(), 1):
            is_injury = 'injury' in row['feature'].lower()
            marker = "🏥" if is_injury else "  "
            print(f"   {marker} {i:2d}. {row['feature']:<35} {row['importance']:.4f}")
        
    def save_models(self):
        """Save hybrid models"""
        print_section_header("Saving NBA Hybrid Models")
        
        models_dir = Path("models/trained_models")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        joblib.dump(self.models['hybrid_ml'], NBA_ML_MODEL_PATH)
        logger.info(f"✅ ML model saved: {NBA_ML_MODEL_PATH}")
        
        joblib.dump(self.models['hybrid_spread'], NBA_SPREAD_MODEL_PATH)
        logger.info(f"✅ Spread model saved: {NBA_SPREAD_MODEL_PATH}")
        
        # Save feature columns
        with open(NBA_FEATURE_COLUMNS_PATH, 'w') as f:
            json.dump(self.feature_columns, f, indent=2)
        logger.info(f"✅ Feature columns saved: {NBA_FEATURE_COLUMNS_PATH}")
        
        # Save metadata
        metadata = {
            'hybrid_games': len(self.hybrid_df),
            'hybrid_features': len(self.hybrid_features),
            'feature_breakdown': {
                'record_features': len([f for f in self.hybrid_features if 'win' in f.lower() or 'record' in f.lower()]),
                'shooting_features': len([f for f in self.hybrid_features if 'fg' in f.lower() or 'shot' in f.lower()]),
                'pace_features': len([f for f in self.hybrid_features if 'pace' in f.lower() or 'poss' in f.lower()]),
                'injury_features': len([f for f in self.hybrid_features if 'injury' in f.lower()]),
                'rest_days': len([f for f in self.hybrid_features if 'rest' in f.lower()])
            },
            'seasons': [int(x) for x in sorted(self.hybrid_df['season'].unique())],
            'models_trained': list(self.models.keys()),
            'includes_injuries': any('injury' in f.lower() for f in self.hybrid_features)
        }
        
        with open(NBA_METADATA_PATH, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✅ Metadata saved: {NBA_METADATA_PATH}")
        
    def run_full_training(self):
        """Run complete hybrid model training"""
        print_section_header("🏀 NBA HYBRID MODEL TRAINING")
        
        self.load_data()
        self.create_hybrid_dataset()
        self.train_hybrid_models()
        self.save_models()
        
        print_section_header("✅ NBA HYBRID MODEL TRAINING COMPLETE!")
        print("Models saved to models/trained_models/")
        print("Ready for predictions!")

