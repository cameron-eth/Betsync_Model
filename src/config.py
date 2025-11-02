"""
Configuration file for BetSync NFL Prediction Model
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" / "trained_models"
SRC_DIR = PROJECT_ROOT / "src"

# Data files
HISTORIC_MATCHUPS_CSV = DATA_DIR / "nfl_historic_matchups.csv"
ENGINEERED_FEATURES_CSV = DATA_DIR / "engineered_features.csv"

# Model files
XGBOOST_MODEL_PATH = MODELS_DIR / "xgboost_model.joblib"
LIGHTGBM_MODEL_PATH = MODELS_DIR / "lightgbm_model.joblib"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://sxhikxboubghkzfurbyh.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InN4aGlreGJvdWJnaGt6ZnVyYnloIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjYwOTc1MzcsImV4cCI6MjA0MTY3MzUzN30.Vcqa7M13EDIzqX8Bw00Lc1oLA8OKcyqeMaVCbOR8lS8')

# Supabase tables
TABLE_TEAM_STATS = "nfl_team_stats"
TABLE_PLAYER_STATS = "nfl_player_stats"
TABLE_PREDICTIONS = "nfl_predictions"
TABLE_MATCHUPS = "nfl_matchups"

# NFL team mappings
NFL_TEAMS = {
    'ARI': 'Arizona Cardinals',
    'ATL': 'Atlanta Falcons',
    'BAL': 'Baltimore Ravens',
    'BUF': 'Buffalo Bills',
    'CAR': 'Carolina Panthers',
    'CHI': 'Chicago Bears',
    'CIN': 'Cincinnati Bengals',
    'CLE': 'Cleveland Browns',
    'DAL': 'Dallas Cowboys',
    'DEN': 'Denver Broncos',
    'DET': 'Detroit Lions',
    'GB': 'Green Bay Packers',
    'HOU': 'Houston Texans',
    'IND': 'Indianapolis Colts',
    'JAX': 'Jacksonville Jaguars',
    'KC': 'Kansas City Chiefs',
    'LV': 'Las Vegas Raiders',
    'LAC': 'Los Angeles Chargers',
    'LAR': 'Los Angeles Rams',
    'MIA': 'Miami Dolphins',
    'MIN': 'Minnesota Vikings',
    'NE': 'New England Patriots',
    'NO': 'New Orleans Saints',
    'NYG': 'New York Giants',
    'NYJ': 'New York Jets',
    'PHI': 'Philadelphia Eagles',
    'PIT': 'Pittsburgh Steelers',
    'SF': 'San Francisco 49ers',
    'SEA': 'Seattle Seahawks',
    'TB': 'Tampa Bay Buccaneers',
    'TEN': 'Tennessee Titans',
    'WAS': 'Washington Commanders'
}

# Reverse mapping: full name to abbreviation
NFL_TEAMS_REVERSE = {v: k for k, v in NFL_TEAMS.items()}

# Model hyperparameters
XGBOOST_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'random_state': 42,
    'tree_method': 'hist'
}

# Advanced model hyperparameters (more conservative to reduce overconfidence)
XGBOOST_ADVANCED_PARAMS = {
    'max_depth': 4,  # Reduced depth
    'learning_rate': 0.03,  # Lower learning rate
    'n_estimators': 150,  # Fewer trees
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'random_state': 42,
    'tree_method': 'hist',
    'reg_alpha': 0.1,  # L1 regularization
    'reg_lambda': 0.1,  # L2 regularization
    'subsample': 0.8,  # Subsample to reduce overfitting
    'colsample_bytree': 0.8  # Feature subsampling
}

LIGHTGBM_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'objective': 'binary',
    'metric': 'binary_logloss',
    'random_state': 42,
    'verbose': -1
}

# Training configuration
TRAIN_TEST_SPLIT = 0.85  # 85% train, 15% test
VALIDATION_SPLIT = 0.15  # 15% of train for validation
RANDOM_STATE = 42
CV_FOLDS = 5

# Feature engineering parameters
LAST_N_GAMES = 5  # Number of recent games for rolling stats
MIN_GAMES_PLAYED = 3  # Minimum games to calculate rolling stats

# Prediction thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.75
MEDIUM_CONFIDENCE_THRESHOLD = 0.60
LOW_CONFIDENCE_THRESHOLD = 0.50

# Logging
LOG_LEVEL = "INFO"

