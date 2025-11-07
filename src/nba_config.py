"""
Configuration file for BetSync NBA Prediction Model
"""

import os
from pathlib import Path

# Project paths (shared with NFL)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" / "trained_models"
SRC_DIR = PROJECT_ROOT / "src"

# NBA Data files
NBA_HISTORIC_MATCHUPS_CSV = DATA_DIR / "nba_historic_matchups.csv"
NBA_ENHANCED_FEATURES_CSV = DATA_DIR / "nba_enhanced_features.csv"
NBA_ENHANCED_FEATURES_CLEAN_CSV = DATA_DIR / "nba_enhanced_features_clean.csv"

# NBA Model files
NBA_ML_MODEL_PATH = MODELS_DIR / "nba_hybrid_ml_model.joblib"
NBA_SPREAD_MODEL_PATH = MODELS_DIR / "nba_hybrid_spread_model.joblib"
NBA_FEATURE_COLUMNS_PATH = MODELS_DIR / "nba_feature_columns.json"
NBA_METADATA_PATH = MODELS_DIR / "nba_metadata.json"

# Supabase credentials (shared)
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://sxhikxboubghkzfurbyh.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InN4aGlreGJvdWJnaGt6ZnVyYnloIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjYwOTc1MzcsImV4cCI6MjA0MTY3MzUzN30.Vcqa7M13EDIzqX8Bw00Lc1oLA8OKcyqeMaVCbOR8lS8')

# Supabase tables (NBA)
NBA_TABLE_TEAM_STATS = "nba_team_stats"
NBA_TABLE_PLAYER_STATS = "nba_player_stats"
NBA_TABLE_PREDICTIONS = "nba_predictions"
NBA_TABLE_MATCHUPS = "nba_matchups"

# NBA team mappings (30 teams)
NBA_TEAMS = {
    'ATL': 'Atlanta Hawks',
    'BOS': 'Boston Celtics',
    'BKN': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets',
    'CHI': 'Chicago Bulls',
    'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks',
    'DEN': 'Denver Nuggets',
    'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors',
    'HOU': 'Houston Rockets',
    'IND': 'Indiana Pacers',
    'LAC': 'LA Clippers',
    'LAL': 'Los Angeles Lakers',
    'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat',
    'MIL': 'Milwaukee Bucks',
    'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans',
    'NYK': 'New York Knicks',
    'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic',
    'PHI': 'Philadelphia 76ers',
    'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers',
    'SAC': 'Sacramento Kings',
    'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors',
    'UTA': 'Utah Jazz',
    'WAS': 'Washington Wizards'
}

# Reverse mapping: full name to abbreviation
NBA_TEAMS_REVERSE = {v: k for k, v in NBA_TEAMS.items()}

# NBA team ID mappings (nba_api uses numeric IDs)
# These will be populated dynamically from nba_api.stats.staticdata.teams
NBA_TEAM_IDS = {}  # Will be loaded from API

# Model hyperparameters (same as NFL)
XGBOOST_ADVANCED_PARAMS = {
    'max_depth': 4,
    'learning_rate': 0.03,
    'n_estimators': 150,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'random_state': 42,
    'tree_method': 'hist',
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8
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
VALIDATION_SPLIT = 0.15
RANDOM_STATE = 42
CV_FOLDS = 5

# Feature engineering parameters
LAST_N_GAMES = 5  # Number of recent games for rolling stats
MIN_GAMES_PLAYED = 3  # Minimum games to calculate rolling stats

# NBA-specific: Rest days threshold (back-to-back games)
BACK_TO_BACK_REST_DAYS = 1  # 1 day between games = back-to-back
REST_DAYS_MIN = 0  # Minimum rest days
REST_DAYS_MAX = 5  # Maximum rest days to consider

# Prediction thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.75
MEDIUM_CONFIDENCE_THRESHOLD = 0.60
LOW_CONFIDENCE_THRESHOLD = 0.50

# Logging
LOG_LEVEL = "INFO"

# NBA seasons (Oct-Apr, season year is year of end date)
# Example: 2023-24 season = 2024, 2025-26 season = 2026
# Using recent seasons for faster testing - can expand to full range later
NBA_SEASONS_AVAILABLE = list(range(2022, 2025))  # 2021-22 through 2023-24 seasons (historical data)
# Current season (2025-26 = 2026) will use live NBA API stats

