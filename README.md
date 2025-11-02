# 🏈 BetSync NFL Prediction Model

Machine learning prediction pipeline for NFL game outcomes. Train on historic matchups, predict live games from Odds API.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Feature Engineering

Process historic matchups and engineer training features:

```bash
cd src
python feature_engineering.py
```

This will:
- Load `data/nfl_historic_matchups.csv` (13,756 games from 1966-2024)
- Calculate last 5 games stats (wins, losses, PPG, home/away splits)
- Calculate season-to-date stats
- Calculate head-to-head matchup history
- Save engineered features to `data/engineered_features.csv`

**Features Generated:**
- Last 5 games: wins, losses, win %, PPG, PPG allowed, point differential
- Last 5 home/away splits: home record, away record
- Season stats: wins, losses, win %, PPG, PPG allowed
- Matchup history: H2H games, wins, avg total points, avg margin
- Context: week, playoff flag, weather, spread, over/under

### 3. Train Models

Train XGBoost and LightGBM models on engineered features:

```bash
python model_training.py
```

This will:
- Load engineered features
- Split data (85% train, 15% test)
- Train XGBoost classifier
- Train LightGBM classifier
- Evaluate with cross-validation
- Compare model performance
- Save models to `models/trained_models/`

**Models Saved:**
- `xgboost_model.joblib` - XGBoost classifier
- `lightgbm_model.joblib` - LightGBM classifier
- `feature_columns.json` - Feature names for inference
- `model_metadata.json` - Training metrics and metadata

### 4. Generate Predictions

Use trained models to predict live games from Odds API:

```bash
python prediction_pipeline.py
```

**Example Usage:**

```python
from prediction_pipeline import NFLPredictionPipeline

# Initialize pipeline
pipeline = NFLPredictionPipeline()
pipeline.load_models()

# Game data from Odds API
game = {
    'home_team': 'Kansas City Chiefs',
    'away_team': 'Buffalo Bills',
    'game_date': '2025-10-20',
    'week': 7,
    'odds': {
        'spread': -3.5,
        'over_under': 54.5
    },
    'weather': {
        'temperature': 72,
        'wind': 8,
        'humidity': 60
    }
}

# Generate prediction
prediction = pipeline.predict_game(game)

# Output:
# {
#     'game_info': {...},
#     'predictions': {
#         'home_win_probability': 0.68,
#         'predicted_winner': 'Kansas City Chiefs',
#         'expected_home_score': 28.5,
#         'expected_away_score': 24.2
#     },
#     'confidence': {
#         'score': 0.82,
#         'label': 'HIGH'
#     }
# }

# Save to Supabase
pipeline.save_prediction_to_db(prediction)
```

## 📊 Model Performance

Models are trained to predict:
1. **Game Outcome** - Home team win (binary classification)
2. **Score Differential** - Point margin (regression)
3. **Total Points** - Combined score (regression)

**Evaluation Metrics:**
- Accuracy
- Precision, Recall, F1 Score
- ROC AUC
- 5-fold Cross-Validation

## 📁 Project Structure

```
betsync_model/
├── data/
│   ├── nfl_historic_matchups.csv     # Historic games (1966-2024)
│   └── engineered_features.csv       # Engineered training data
├── models/
│   └── trained_models/
│       ├── xgboost_model.joblib      # Trained XGBoost
│       ├── lightgbm_model.joblib     # Trained LightGBM
│       ├── feature_columns.json      # Feature names
│       └── model_metadata.json       # Training metadata
├── src/
│   ├── config.py                     # Configuration
│   ├── utils.py                      # Helper functions
│   ├── feature_engineering.py        # Feature pipeline
│   ├── model_training.py             # Training pipeline
│   └── prediction_pipeline.py        # Prediction pipeline
└── requirements.txt
```

## 🔧 Configuration

Edit `src/config.py` to customize:

```python
# Supabase
SUPABASE_URL = "your-url"
SUPABASE_KEY = "your-key"

# Model parameters
XGBOOST_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 200
}

# Feature engineering
LAST_N_GAMES = 5  # Rolling window size
MIN_GAMES_PLAYED = 3  # Min games for stats
```

## 📈 Feature Engineering Details

### Last N Games (Sliding Window)

For each game, calculate stats based on previous N games:

**Overall Performance:**
- `last5_wins`, `last5_losses`
- `last5_win_pct`
- `last5_ppg` (points per game)
- `last5_ppg_allowed`
- `last5_point_diff`

**Home/Away Splits:**
- `last5_home_wins`, `last5_home_losses`, `last5_home_record`
- `last5_away_wins`, `last5_away_losses`, `last5_away_record`

**Rest:**
- `rest_days` (days since last game)

### Season Stats

Cumulative season performance:
- `season_wins`, `season_losses`, `season_win_pct`
- `season_ppg`, `season_ppg_allowed`
- `season_games_played`

### Matchup History

Head-to-head between teams:
- `h2h_games_played`
- `h2h_home_wins`, `h2h_away_wins`
- `h2h_avg_total_points`
- `h2h_avg_margin`

### Contextual

- `week`, `season`, `is_playoff`
- `weather_temp`, `weather_wind`, `weather_humidity`
- `spread`, `over_under`
- `is_neutral_site`

## 🎯 Prediction Confidence

Confidence scores are calculated based on:
- **Decisiveness** - How far from 50/50 (0.5)
- **Data Completeness** - % of features available

**Confidence Levels:**
- HIGH: ≥75%
- MEDIUM: 60-75%
- LOW: 50-60%
- VERY_LOW: <50%

## 🗄️ Database Schema

### `nfl_predictions` Table

```sql
CREATE TABLE nfl_predictions (
    id BIGSERIAL PRIMARY KEY,
    home_team VARCHAR(255),
    away_team VARCHAR(255),
    game_date DATE,
    home_win_probability FLOAT,
    away_win_probability FLOAT,
    predicted_winner VARCHAR(255),
    expected_home_score FLOAT,
    expected_away_score FLOAT,
    expected_total FLOAT,
    confidence_score FLOAT,
    confidence_label VARCHAR(50),
    prediction_timestamp TIMESTAMP,
    model_version VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);
```

## 🔄 Workflow

### One-Time Setup:
1. Move historic CSV to `data/`
2. Run feature engineering → creates training dataset
3. Run model training → trains and saves models

### Production Use:
1. Fetch upcoming games from Odds API
2. Run prediction pipeline with game data
3. Predictions stored in Supabase
4. Frontend displays predictions

### Continuous Improvement:
1. After games complete, collect actual results
2. Calculate prediction accuracy
3. Retrain model periodically with new data

## 🏆 Model Updates

To retrain with new data:

```bash
# Add new games to nfl_historic_matchups.csv

# Re-run feature engineering
python src/feature_engineering.py

# Re-train models
python src/model_training.py

# Models automatically versioned by timestamp
```

## 🐛 Troubleshooting

### Issue: "File not found" error
**Solution:** Ensure you've run feature engineering first to create `engineered_features.csv`

### Issue: Model predictions seem off
**Solution:** Check if recent games data is available in Supabase for feature calculation

### Issue: Low confidence scores
**Solution:** This is normal when teams have limited recent game data (early season)

## 📝 Notes

- Models train on 13,756 historic games (1966-2024)
- Predictions require recent game data (last 5 games) for accuracy
- Confidence scores reflect both prediction decisiveness and data availability
- Ensemble of XGBoost + LightGBM for robust predictions

## 🔮 Future Enhancements

- Add player injury impact modeling
- Incorporate betting market movements
- Add weather impact analysis
- Implement neural networks for deep learning
- Add explainability (SHAP values)
- Real-time model updating

---

**Built with:** XGBoost, LightGBM, Scikit-learn, Pandas, Supabase




# Betsync_Model
