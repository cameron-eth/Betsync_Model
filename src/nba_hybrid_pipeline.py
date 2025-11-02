"""
NBA Hybrid Model Pipeline
Streamlined pipeline for NBA predictions
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import os
import requests
from supabase import create_client, Client
from datetime import datetime

try:
    from .nba_config import (
        SUPABASE_URL, SUPABASE_KEY, NBA_TABLE_PREDICTIONS,
        NBA_ML_MODEL_PATH, NBA_SPREAD_MODEL_PATH, NBA_FEATURE_COLUMNS_PATH
    )
    from .nba_advanced_features import NBAAdvancedFeatureEngine
    from .nba_utils import (
        print_section_header, normalize_team_name, get_team_abbr,
        calculate_confidence_score, get_confidence_label, safe_float
    )
except ImportError:
    from nba_config import (
        SUPABASE_URL, SUPABASE_KEY, NBA_TABLE_PREDICTIONS,
        NBA_ML_MODEL_PATH, NBA_SPREAD_MODEL_PATH, NBA_FEATURE_COLUMNS_PATH
    )
    from nba_advanced_features import NBAAdvancedFeatureEngine
    from nba_utils import (
        print_section_header, normalize_team_name, get_team_abbr,
        calculate_confidence_score, get_confidence_label, safe_float
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NBAHybridPipeline:
    """NBA hybrid model pipeline"""
    
    def __init__(self, current_season: int = 2025):
        self.hybrid_ml_model = None
        self.hybrid_spread_model = None
        self.feature_columns = None
        self.supabase = None
        self.current_season = current_season
        self.feature_engine = None
        
        # Initialize Supabase
        self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
    def load_hybrid_model(self):
        """Load the hybrid model"""
        print_section_header("Loading NBA Hybrid Model")
        
        models_dir = Path("models/trained_models")
        
        # Load models
        self.hybrid_ml_model = joblib.load(NBA_ML_MODEL_PATH)
        self.hybrid_spread_model = joblib.load(NBA_SPREAD_MODEL_PATH)
        
        # Load feature columns
        with open(NBA_FEATURE_COLUMNS_PATH, 'r') as f:
            self.feature_columns = json.load(f)['hybrid']
        
        # Initialize feature engine
        self.feature_engine = NBAAdvancedFeatureEngine(seasons=[self.current_season])
        self.feature_engine.load_all_data()
        
        print(f"✅ Loaded hybrid ML model")
        print(f"✅ Loaded hybrid Spread model")
        print(f"✅ Loaded {len(self.feature_columns)} feature columns")
        
    def get_upcoming_games(self):
        """Get upcoming games from Odds API"""
        print_section_header("Loading Upcoming Games from Odds API")
        
        api_key = os.getenv('ODDS_API_KEY')
        if not api_key:
            raise ValueError("❌ ODDS_API_KEY environment variable is required")
        
        api_key = api_key.strip()
        
        url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds/?apiKey={api_key}&regions=us&markets=h2h,spreads"
        
        print(f"🔍 Fetching games from Odds API...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        games_data = response.json()
        print(f"✅ Fetched {len(games_data)} games from Odds API")
        
        if len(games_data) == 0:
            raise ValueError("❌ No games found from Odds API")
        
        # Process games
        upcoming_games = []
        for game in games_data:
            spread = self.extract_spread(game.get('bookmakers', []), game.get('home_team'))
            home_ml_odds, away_ml_odds = self.extract_ml_odds(
                game.get('bookmakers', []), 
                game.get('home_team'),
                game.get('away_team')
            )
            over_under = self.extract_over_under(game.get('bookmakers', []))
            
            game_data = {
                'id': game['id'],
                'away_team': game['away_team'],
                'home_team': game['home_team'],
                'commence_time': game['commence_time'],
                'spread': spread,
                'home_ml_odds': home_ml_odds,
                'away_ml_odds': away_ml_odds,
                'over_under': over_under,
                'season': self.current_season
            }
            upcoming_games.append(game_data)
        
        print(f"✅ Processed {len(upcoming_games)} upcoming games")
        return upcoming_games
    
    def extract_spread(self, bookmakers, home_team):
        """Extract consensus spread from bookmakers"""
        home_spreads = []
        for bookmaker in bookmakers:
            for market in bookmaker.get('markets', []):
                if market.get('key') == 'spreads':
                    for outcome in market.get('outcomes', []):
                        if outcome.get('name') == home_team and 'point' in outcome:
                            point = outcome['point']
                            if point is not None:
                                home_spreads.append(point)
        
        if home_spreads:
            return round(sum(home_spreads) / len(home_spreads), 1)
        return None
    
    def decimal_to_american(self, decimal_odds):
        """Convert decimal odds to American odds"""
        if decimal_odds == 1.0:
            return 100
        elif decimal_odds >= 2.0:
            return round((decimal_odds - 1) * 100)
        else:
            return round(-100 / (decimal_odds - 1))
    
    def american_to_implied_prob(self, odds):
        """Convert American odds to implied probability"""
        if odds is None:
            return None
        try:
            odds = float(odds)
        except (TypeError, ValueError):
            return None
        if odds >= 0:
            return 100.0 / (odds + 100.0)
        else:
            return -odds / (-odds + 100.0)
    
    def extract_ml_odds(self, bookmakers, home_team, away_team):
        """Extract consensus ML odds from bookmakers"""
        home_odds = []
        away_odds = []
        
        for bookmaker in bookmakers:
            for market in bookmaker.get('markets', []):
                if market.get('key') == 'h2h':
                    for outcome in market.get('outcomes', []):
                        if outcome.get('name') == home_team:
                            if 'price' in outcome:
                                american_odds = self.decimal_to_american(outcome['price'])
                                home_odds.append(american_odds)
                        elif outcome.get('name') == away_team:
                            if 'price' in outcome:
                                american_odds = self.decimal_to_american(outcome['price'])
                                away_odds.append(american_odds)
        
        home_avg = round(sum(home_odds) / len(home_odds)) if home_odds else None
        away_avg = round(sum(away_odds) / len(away_odds)) if away_odds else None
        
        return home_avg, away_avg
    
    def extract_over_under(self, bookmakers):
        """Extract consensus over/under from bookmakers"""
        totals = []
        
        for bookmaker in bookmakers:
            for market in bookmaker.get('markets', []):
                if market.get('key') == 'totals':
                    for outcome in market.get('outcomes', []):
                        if 'point' in outcome and outcome['point'] is not None:
                            totals.append(outcome['point'])
        
        return round(sum(totals) / len(totals), 1) if totals else None
    
    def create_game_features(self, game: Dict) -> pd.DataFrame:
        """Create features for a game"""
        home_team = normalize_team_name(game['home_team'])
        away_team = normalize_team_name(game['away_team'])
        game_date = game['commence_time'][:10]  # Extract date
        
        home_abbr = get_team_abbr(home_team)
        away_abbr = get_team_abbr(away_team)
        
        if not home_abbr or not away_abbr:
            logger.warning(f"Could not find team abbreviations for {home_team} vs {away_team}")
            return None
        
        # Calculate team features
        home_features = self.feature_engine.calculate_team_features(
            home_abbr, self.current_season, game_date
        )
        away_features = self.feature_engine.calculate_team_features(
            away_abbr, self.current_season, game_date
        )
        
        # Combine features
        features = {}
        for key, value in home_features.items():
            features[f'home_{key}'] = value
        for key, value in away_features.items():
            features[f'away_{key}'] = value
        
        # Add differentials
        features['ppg_diff'] = home_features.get('ppg', 0) - away_features.get('ppg', 0)
        features['win_pct_diff'] = home_features.get('win_pct', 0.5) - away_features.get('win_pct', 0.5)
        features['rest_days_diff'] = home_features.get('rest_days', 2) - away_features.get('rest_days', 2)
        
        # Add spread if available
        if game.get('spread'):
            features['spread'] = game['spread']
        else:
            features['spread'] = 0.0
        
        # Convert to DataFrame
        feature_df = pd.DataFrame([features])
        
        # Select only model features
        available_features = [f for f in self.feature_columns if f in feature_df.columns]
        missing_features = [f for f in self.feature_columns if f not in feature_df.columns]
        
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features, filling with 0")
            for f in missing_features:
                feature_df[f] = 0.0
        
        return feature_df[self.feature_columns]
    
    def predict_game(self, game: Dict) -> Dict:
        """Predict a single game"""
        try:
            features = self.create_game_features(game)
            if features is None:
                return None
            
            # Fill NaN values
            features = features.fillna(0)
            
            # Predict
            ml_prob = self.hybrid_ml_model.predict_proba(features)[0][1]
            spread_prob = self.hybrid_spread_model.predict_proba(features)[0][1]
            
            # Calculate confidence
            confidence = calculate_confidence_score(ml_prob, {}, 1.0)
            confidence_label = get_confidence_label(confidence)
            
            # Estimate scores (simplified)
            avg_ppg = 110  # NBA average
            home_score = avg_ppg + (ml_prob - 0.5) * 10
            away_score = avg_ppg - (ml_prob - 0.5) * 10
            
            prediction = {
                'game_info': {
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'game_date': game['commence_time'][:10],
                    'spread': game.get('spread'),
                    'over_under': game.get('over_under')
                },
                'predictions': {
                    'home_win_probability': float(ml_prob),
                    'away_win_probability': float(1 - ml_prob),
                    'predicted_winner': game['home_team'] if ml_prob > 0.5 else game['away_team'],
                    'expected_home_score': float(home_score),
                    'expected_away_score': float(away_score),
                    'expected_total': float(home_score + away_score),
                    'spread_covers_probability': float(spread_prob)
                },
                'confidence': {
                    'score': float(confidence),
                    'label': confidence_label
                }
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting game: {e}")
            return None
    
    def save_prediction_to_db(self, prediction: Dict):
        """Save prediction to Supabase using upsert to avoid duplicates"""
        try:
            game_info = prediction['game_info']
            preds = prediction['predictions']
            conf = prediction['confidence']
            
            data = {
                'home_team': game_info['home_team'],
                'away_team': game_info['away_team'],
                'game_date': game_info['game_date'],
                'home_win_probability': preds['home_win_probability'],
                'away_win_probability': preds['away_win_probability'],
                'predicted_winner': preds['predicted_winner'],
                'expected_home_score': preds['expected_home_score'],
                'expected_away_score': preds['expected_away_score'],
                'expected_total': preds['expected_total'],
                'confidence_score': conf['score'],
                'confidence_label': conf['label'],
                'prediction_timestamp': datetime.now().isoformat(),
                'model_version': 'nba_hybrid_v1'
            }
            
            # Check if prediction already exists
            existing = self.supabase.table(NBA_TABLE_PREDICTIONS).select('id').eq(
                'home_team', game_info['home_team']
            ).eq(
                'away_team', game_info['away_team']
            ).eq(
                'game_date', game_info['game_date']
            ).execute()
            
            if existing.data and len(existing.data) > 0:
                # Update existing prediction
                self.supabase.table(NBA_TABLE_PREDICTIONS).update(data).eq(
                    'id', existing.data[0]['id']
                ).execute()
                logger.info(f"✅ Updated prediction for {game_info['home_team']} vs {game_info['away_team']}")
            else:
                # Insert new prediction
                self.supabase.table(NBA_TABLE_PREDICTIONS).insert(data).execute()
                logger.info(f"✅ Saved prediction for {game_info['home_team']} vs {game_info['away_team']}")
            
        except Exception as e:
            logger.error(f"Error saving prediction to DB: {e}")
    
    def run_full_pipeline(self):
        """Run complete prediction pipeline"""
        print_section_header("🏀 NBA HYBRID MODEL PREDICTIONS")
        
        # Load model
        self.load_hybrid_model()
        
        # Get upcoming games
        games = self.get_upcoming_games()
        
        # Make predictions
        print_section_header("Making Predictions")
        
        predictions_made = 0
        for game in games:
            prediction = self.predict_game(game)
            if prediction:
                print(f"\n📊 {game['away_team']} @ {game['home_team']}")
                print(f"   Home Win Probability: {prediction['predictions']['home_win_probability']:.1%}")
                print(f"   Predicted Winner: {prediction['predictions']['predicted_winner']}")
                print(f"   Confidence: {prediction['confidence']['label']}")
                
                # Save to database
                self.save_prediction_to_db(prediction)
                predictions_made += 1
        
        print_section_header("✅ PREDICTIONS COMPLETE")
        print(f"Made {predictions_made} predictions")
        print("Predictions saved to Supabase")

