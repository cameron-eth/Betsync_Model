# Betsync_Model

NFL and NBA sports prediction models using machine learning.

## 📊 Model Performance

| Sport | ML Accuracy | Spread Accuracy | Status |
|-------|-------------|-----------------|--------|
| **NFL 🏈** | 61.4% | **69.3%** | ✅ Production |
| **NBA 🏀** | 64.2% | 63.8% | ✅ Production |

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Supabase account (for storing predictions)
- Odds API key (for fetching game odds)

### Installation

```bash
# Clone the repository
git clone https://github.com/cameron-eth/Betsync_Model.git
cd Betsync_Model

# Install dependencies
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env
# Edit .env with your credentials
```

### Running Predictions Locally

```bash
# NFL predictions
python run_pipeline.py --predict

# NBA predictions  
python run_nba_pipeline.py --predict
```

## 🤖 GitHub Actions (Automated Predictions)

### Setup Secrets

Add these secrets in **Settings → Secrets → Actions**:
- `SUPABASE_URL` - Your Supabase project URL
- `SUPABASE_KEY` - Your Supabase anon/service key
- `ODDS_API_KEY` - Your Odds API key

### Automated Schedule

**Predictions run automatically 2x daily:**
- 6:00 AM EST (11:00 UTC)
- 12:00 PM EST (17:00 UTC)

**Manual triggers available:**
- Go to **Actions** tab
- Select "Run NFL and NBA Predictions"
- Click "Run workflow"

## 📁 Repository Structure

```
betsync_model/
├── .github/workflows/       # GitHub Actions
│   ├── run-predictions.yml  # Daily predictions (automated)
│   └── retrain-models.yml   # Model retraining (manual only)
├── data/                    # Training data
│   ├── nfl_enhanced_features_v2.csv     # NFL data (3.4M)
│   ├── nba_enhanced_features_clean.csv  # NBA data (3.2M)
│   └── nfl_features_to_keep.csv         # Feature selection
├── models/trained_models/   # Pre-trained models
│   ├── nfl_final_ml_model.joblib        # NFL ML model
│   ├── nfl_final_spread_model.joblib    # NFL Spread model
│   ├── nba_hybrid_ml_model.joblib       # NBA ML model
│   └── nba_hybrid_spread_model.joblib   # NBA Spread model
├── src/                     # Source code
│   ├── pure_hybrid_pipeline.py          # NFL pipeline
│   ├── nba_hybrid_pipeline.py           # NBA pipeline
│   └── ...
├── run_pipeline.py          # NFL CLI
├── run_nba_pipeline.py      # NBA CLI
└── requirements.txt         # Dependencies
```

## 🏈 NFL Model Details

**Performance:**
- ML Accuracy: 61.4% (improved from 57.7%)
- Spread Accuracy: 69.3% (improved from 60.9%)
- Overfitting: 14.5% gap (ML), 10.5% gap (Spread)

**Improvements Made:**
1. **Phase 1 - Cleanup:** Removed 41 problematic features
   - Fixed data leakage (3 missing market odds features)
   - Removed 18 noisy low-signal features
   - Result: +3.2% ML, +8.2% Spread

2. **Phase 2 - Feature Engineering:** Added powerful features
   - Rolling win percentage (0.18 correlation)
   - EPA differentials (0.23 correlation - strongest!)
   - Playoff push indicators

3. **Phase 3 - Hyperparameter Tuning:** Controlled overfitting
   - Aggressive regularization (max_depth=4, gamma=0.3)
   - Reduced overfitting from 32% → 14.5%

**Key Features (95 total):**
- Record-based: win%, season record, rolling win%
- EPA stats: offensive/defensive efficiency
- Situational: rest days, playoff implications
- Home/away splits
- Injury impact (refined, position-weighted)

## 🏀 NBA Model Details

**Performance:**
- ML Accuracy: 64.2%
- Spread Accuracy: 63.8%
- Dataset: 1,230 games (2024 season)

**Key Features (257 total):**
- Win percentage (season, home, away, last 5 games)
- Shooting stats (FG%, 3P%, FT%, TS%, eFG%)
- Rebounds, assists, turnovers
- Rest days and schedule
- Home/away splits
- Pace and efficiency ratings

**Data Source:** `nba_api` library (real-time NBA.com data)

## 🔧 Advanced Usage

### Model Training (Local Only)

**Note:** Training requires historical data not available in GitHub Actions.

```bash
# Generate features (requires data sources)
python run_pipeline.py --features

# Train models
python run_pipeline.py --train

# NBA equivalent
python run_nba_pipeline.py --features
python run_nba_pipeline.py --train
```

### Feature Analysis

```bash
# Check feature importance
python show_feature_importance.py

# Run diagnostics
python -c "from src.nba_config import *; print(NBA_SEASONS_AVAILABLE)"
```

## 📊 Database Schema

### NFL Predictions Table
```sql
CREATE TABLE predictions (
    id BIGSERIAL PRIMARY KEY,
    home_team VARCHAR(255),
    away_team VARCHAR(255),
    game_date DATE,
    home_win_probability FLOAT,
    predicted_winner VARCHAR(255),
    expected_home_score FLOAT,
    expected_away_score FLOAT,
    confidence_score FLOAT,
    prediction_timestamp TIMESTAMP
);
```

### NBA Predictions Table
```sql
CREATE TABLE nba_predictions (
    id BIGSERIAL PRIMARY KEY,
    home_team VARCHAR(255),
    away_team VARCHAR(255),
    game_date DATE,
    home_win_probability FLOAT,
    predicted_winner VARCHAR(255),
    expected_home_score FLOAT,
    expected_away_score FLOAT,
    confidence_score FLOAT,
    prediction_timestamp TIMESTAMP,
    UNIQUE(home_team, away_team, game_date)
);
```

## 🤝 Contributing

Models are production-ready. For improvements:
1. Fork the repository
2. Create a feature branch
3. Test locally with your own data
4. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- NFL data: nflfastR, ESPN
- NBA data: nba_api (NBA.com official stats)
- Odds data: The Odds API
- Machine Learning: XGBoost, scikit-learn

---

**Built with ❤️ for accurate sports predictions**
