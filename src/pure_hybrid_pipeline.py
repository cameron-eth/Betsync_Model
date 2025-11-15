"""
Pure Hybrid Model Pipeline
Streamlined pipeline that runs all dependencies then the pure hybrid model for Week 7 predictions
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
import subprocess
import sys
import os
import requests
from supabase import create_client, Client

# Import config and feature fetcher
try:
    from .config import XGBOOST_ADVANCED_PARAMS
    from .current_season_features import CurrentSeasonFeatures
except ImportError:
    from config import XGBOOST_ADVANCED_PARAMS
    from current_season_features import CurrentSeasonFeatures

def print_section_header(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*80}")
    print(f"🏈 {title}")
    print(f"{'='*80}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PureHybridPipeline:
    """Pure hybrid model pipeline - THE model"""
    
    def __init__(self, current_season: int = 2025, current_week: int = 8):
        self.hybrid_ml_model = None
        self.hybrid_spread_model = None
        self.feature_columns = None
        self.historical_df = None
        self.supabase = None
        self.current_season = current_season
        self.current_week = current_week
        self.feature_fetcher = None
        
        # Initialize Supabase if credentials are available
        if os.getenv('SUPABASE_URL') and os.getenv('SUPABASE_KEY'):
            self.supabase = create_client(
                os.getenv('SUPABASE_URL'),
                os.getenv('SUPABASE_KEY')
            )
        else:
            # Use hardcoded credentials from config
            from src.config import SUPABASE_URL, SUPABASE_KEY
            self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
    def run_dependencies(self):
        """Run all dependencies before hybrid model"""
        print_section_header("Running Dependencies")
        
        # 1. Generate advanced features (if needed)
        print("1️⃣ Generating Advanced Features...")
        try:
            result = subprocess.run([sys.executable, "-m", "src.advanced_features"], 
                                  capture_output=True, text=True, cwd=".")
            if result.returncode == 0:
                print("✅ Advanced features generated successfully")
            else:
                print(f"⚠️ Advanced features generation had issues: {result.stderr}")
        except Exception as e:
            print(f"⚠️ Advanced features generation failed: {e}")
        
        # 2. Train hybrid model (if needed)
        print("\n2️⃣ Training Hybrid Model...")
        try:
            result = subprocess.run([sys.executable, "-m", "src.true_hybrid_model"], 
                                  capture_output=True, text=True, cwd=".")
            if result.returncode == 0:
                print("✅ Hybrid model trained successfully")
            else:
                print(f"⚠️ Hybrid model training had issues: {result.stderr}")
        except Exception as e:
            print(f"⚠️ Hybrid model training failed: {e}")
        
        print("✅ All dependencies completed")
        
    def load_hybrid_model(self):
        """Load the pure hybrid model"""
        print_section_header("Loading Pure Hybrid Model & Current Season Features")
        
        models_dir = Path("models/trained_models")
        
        # Load models
        self.hybrid_ml_model = joblib.load(models_dir / "true_hybrid_hybrid_ml_model.joblib")
        self.hybrid_spread_model = joblib.load(models_dir / "true_hybrid_hybrid_spread_model.joblib")
        
        # Load feature columns
        with open(models_dir / "true_hybrid_feature_columns.json", 'r') as f:
            self.feature_columns = json.load(f)['hybrid']
        
        # Load historical data for weather/stadium lookup
        self.historical_df = pd.read_csv("data/nfl_historic_matchups.csv")
        
        # Load current season features using nflreadpy
        print(f"\n📊 Loading {self.current_season} season features (up to week {self.current_week - 1})...")
        self.feature_fetcher = CurrentSeasonFeatures(
            current_season=self.current_season,
            current_week=self.current_week
        )
        team_features = self.feature_fetcher.load_and_calculate_features()
        
        if team_features is None:
            logger.warning("⚠️  Could not load current season features, predictions may be less accurate")
        
        print(f"✅ Loaded hybrid ML model (calibrated with isotonic regression)")
        print(f"✅ Loaded hybrid Spread model (calibrated with isotonic regression)")
        print(f"✅ Loaded {len(self.feature_columns)} feature columns")
        print(f"✅ Loaded historical data for weather/stadium lookup")
        if team_features is not None:
            print(f"✅ Loaded {self.current_season} season features for {len(team_features)} teams")
        
    def get_current_week_games(self):
        """Get current week games from Odds API"""
        print_section_header("Loading Current Week Games from Odds API")
        
        # Fetch current week NFL games from Odds API
        api_key = os.getenv('ODDS_API_KEY')
        if not api_key:
            raise ValueError("❌ ODDS_API_KEY environment variable is required")
        
        # Clean the API key (remove any newlines or whitespace)
        api_key = api_key.strip()
        
        url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/?apiKey={api_key}&regions=us&markets=h2h,spreads"
        
        print(f"🔍 Fetching games from Odds API...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        games_data = response.json()
        print(f"✅ Fetched {len(games_data)} games from Odds API")
        
        if len(games_data) == 0:
            raise ValueError("❌ No games found from Odds API")
        
        # Convert to our format and filter for upcoming week games
        current_week_games = []
        current_week = self.get_current_nfl_week()
        
        for game in games_data:
            game_week = self.calculate_nfl_week(game['commence_time'])
            
            # Process games for current week and next week (upcoming games)
            if game_week >= current_week:
                # Extract odds data from bookmakers
                spread = self.extract_spread(game.get('bookmakers', []), game.get('home_team'))
                home_ml_odds, away_ml_odds = self.extract_ml_odds(
                    game.get('bookmakers', []), 
                    game.get('home_team'),
                    game.get('away_team')
                )
                over_under = self.extract_over_under(game.get('bookmakers', []))
                
                market_home_prob = None
                market_away_prob = None

                if home_ml_odds is not None and away_ml_odds is not None:
                    home_implied = self.american_to_implied_prob(home_ml_odds)
                    away_implied = self.american_to_implied_prob(away_ml_odds)

                    if home_implied is not None and away_implied is not None:
                        total = home_implied + away_implied
                        if total > 0:
                            market_home_prob = home_implied / total
                            market_away_prob = away_implied / total

                game_data = {
                    'id': game['id'],
                    'away_team': game['away_team'],
                    'home_team': game['home_team'],
                    'commence_time': game['commence_time'],
                    'spread': spread,
                    'home_ml_odds': home_ml_odds,
                    'away_ml_odds': away_ml_odds,
                    'over_under': over_under,
                    'market_home_ml_prob': market_home_prob,
                    'market_away_ml_prob': market_away_prob,
                    'market_prob_diff': (market_home_prob - market_away_prob)
                    if market_home_prob is not None and market_away_prob is not None
                    else None,
                    'week': game_week,
                    'season': 2025
                }
                current_week_games.append(game_data)
        
        print(f"✅ Processed {len(current_week_games)} current week games")
        return current_week_games
    
    def extract_spread(self, bookmakers, home_team):
        """Extract consensus spread from bookmakers for home team"""
        home_spreads = []
        for bookmaker in bookmakers:
            for market in bookmaker.get('markets', []):
                if market.get('key') == 'spreads':
                    for outcome in market.get('outcomes', []):
                        # Match the home team specifically
                        if outcome.get('name') == home_team and 'point' in outcome:
                            point = outcome['point']
                            if point is not None:
                                home_spreads.append(point)
        
        # Return consensus spread (average of home team spreads)
        if home_spreads:
            return round(sum(home_spreads) / len(home_spreads), 1)
        
        # If no spreads found, return None to indicate missing data
        return None
    
    def decimal_to_american(self, decimal_odds):
        """Convert decimal odds to American odds"""
        if decimal_odds == 1.0:
            # Even odds (very rare)
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
                                # Convert from decimal to American odds
                                american_odds = self.decimal_to_american(outcome['price'])
                                home_odds.append(american_odds)
                        elif outcome.get('name') == away_team:
                            if 'price' in outcome:
                                # Convert from decimal to American odds
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
    
    def calculate_nfl_week(self, commence_time):
        """Calculate NFL week from game date"""
        from datetime import datetime, timezone
        game_date = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
        
        # For 2025 season, Week 1 starts September 4th
        # Calculate which week this game falls into
        season_start = datetime(2025, 9, 4, tzinfo=timezone.utc)  # Week 1 start
        days_since_start = (game_date - season_start).days
        
        # NFL weeks run Thursday to Wednesday
        # Week 1: Sep 4-10, Week 2: Sep 11-17, etc.
        week = max(1, min(18, (days_since_start // 7) + 1))
        
        return week
    
    def get_current_nfl_week(self):
        """Get the current NFL week based on today's date"""
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc)
        
        # For 2025 season, Week 1 starts September 4th
        season_start = datetime(2025, 9, 4, tzinfo=timezone.utc)
        days_since_start = (today - season_start).days
        
        # NFL weeks run Thursday to Wednesday
        week = max(1, min(18, (days_since_start // 7) + 1))
        
        return week
    
    def normalize_team_name_for_api(self, team_name: str) -> str:
        """Convert full team names from Odds API to abbreviations for feature lookup"""
        team_mapping = {
            'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
            'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
            'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
            'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
            'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
            'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
            'Los Angeles Rams': 'LAR', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
            'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
            'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
            'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
            'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS'
        }
        return team_mapping.get(team_name, team_name)
    
    def create_game_feature_vector(self, game):
        """Create a feature vector for a specific game using REAL current season features"""
        if self.feature_fetcher is None or self.feature_fetcher.team_features is None:
            logger.warning("No current season features available, using defaults")
            return [0.0] * len(self.feature_columns)
        
        # Convert team names to abbreviations
        home_team = self.normalize_team_name_for_api(game['home_team'])
        away_team = self.normalize_team_name_for_api(game['away_team'])
        spread_value = game.get('spread') if game.get('spread') is not None else 0.0
        week = game.get('week', self.current_week)
        
        # Get real matchup features from current season data
        market_home_prob = game.get('home_ml_odds_implied')
        market_away_prob = game.get('away_ml_odds_implied')
        
        matchup_features = self.feature_fetcher.create_matchup_features(
            home_team, away_team, spread_value, week,
            market_home_ml_prob=market_home_prob,
            market_away_ml_prob=market_away_prob
        )
        
        # Map features to model's expected feature order
        # The model was trained on specific features, so we need to match them
        feature_vector = []
        for col in self.feature_columns:
            if col in matchup_features:
                feature_vector.append(matchup_features[col])
            else:
                # Feature not available, use default
                logger.warning(f"Feature {col} not found in matchup features, using 0.0")
                feature_vector.append(0.0)
        
        return feature_vector
    
    def calculate_edge_and_quality(self, our_prob, home_ml_odds, away_ml_odds):
        """Calculate edge percentage and bet quality based on our probability vs implied odds"""
        # For spread betting, we need to calculate the actual implied probability
        # If we have ML odds, we can derive the implied probability from those
        if home_ml_odds and away_ml_odds:
            # Convert American odds to implied probabilities
            # Use american_to_implied_prob() because odds from API are in normal format (not basis points)
            home_implied = self.american_to_implied_prob(home_ml_odds)
            away_implied = self.american_to_implied_prob(away_ml_odds)
            
            # For spread betting, use the average or the closer to 50%
            implied_prob = (home_implied + away_implied) / 2
        else:
            # If no odds available, don't calculate edge - it's meaningless
            # Return neutral values instead of creating false edges
            return 0.0, "NO DATA", "NO BET"
        
        # Calculate edge
        edge = (our_prob - implied_prob) * 100
        
        # Realistic quality thresholds for 3-10% edge range
        if edge >= 8:
            bet_quality = "EXCELLENT"
        elif edge >= 5:
            bet_quality = "VERY GOOD"
        elif edge >= 3:
            bet_quality = "GOOD"
        elif edge >= 1:
            bet_quality = "FAIR"
        else:
            bet_quality = "POOR"
        
        # Determine recommended bet - only recommend if we have real edge
        if edge >= 3:
            if our_prob > implied_prob:
                recommended_bet = "HOME"
            else:
                recommended_bet = "AWAY"
        else:
            recommended_bet = "-"
        
        return round(edge, 1), bet_quality, recommended_bet
    
    def american_odds_to_prob(self, odds):
        """Convert American odds to implied probability"""
        if odds is None:
            return None
        
        # The odds are stored in database as integers * 100 (e.g., -42100 instead of -421)
        # Convert back to actual odds value
        actual_odds = odds / 100.0
        
        if actual_odds > 0:
            return 100 / (actual_odds + 100)
        else:
            return abs(actual_odds) / (abs(actual_odds) + 100)
    
    def calculate_ml_edges(self, home_win_prob, away_win_prob, home_ml_odds, away_ml_odds):
        """Calculate ML edges for both teams"""
        home_ml_edge = None
        away_ml_edge = None
        
        if home_ml_odds is not None:
            # Use american_to_implied_prob() because odds from API are in normal format (not basis points)
            home_implied = self.american_to_implied_prob(home_ml_odds)
            if home_implied:
                home_ml_edge = (home_win_prob - home_implied) * 100
        
        if away_ml_odds is not None:
            # Use american_to_implied_prob() because odds from API are in normal format (not basis points)
            away_implied = self.american_to_implied_prob(away_ml_odds)
            if away_implied:
                away_ml_edge = (away_win_prob - away_implied) * 100
        
        return home_ml_edge, away_ml_edge
    
    def get_confidence_level(self, probability):
        """Get confidence level based on probability - more conservative thresholds"""
        # More conservative confidence levels
        if probability >= 0.85 or probability <= 0.15:
            return "HIGH"
        elif probability >= 0.70 or probability <= 0.30:
            return "MEDIUM"
        else:
            return "LOW"
    
    def validate_probability(self, prob, prob_type="win"):
        """Validate and cap unrealistic probabilities"""
        if prob is None:
            return 0.5
        
        # Cap extreme probabilities that are likely model errors
        if prob > 0.95:
            return 0.95
        elif prob < 0.05:
            return 0.05
        
        return prob
    
    def lookup_stadium_weather(self, home_team, away_team):
        """Look up stadium and weather from historical data"""
        if self.historical_df is None:
            return 'Unknown', 'Unknown'
        
        # Try to find a recent match between these teams at home stadium
        matches = self.historical_df[
            (self.historical_df['team_home'] == home_team) & 
            (self.historical_df['team_away'] == away_team)
        ]
        
        if len(matches) > 0:
            # Get most recent match
            recent_match = matches.iloc[-1]
            stadium = recent_match.get('stadium', 'Unknown')
            weather = recent_match.get('weather_detail', 'Unknown')
            return stadium, weather
        
        # If no exact match, try just home team matches
        home_matches = self.historical_df[self.historical_df['team_home'] == home_team]
        if len(home_matches) > 0:
            recent_match = home_matches.iloc[-1]
            stadium = recent_match.get('stadium', 'Unknown')
            # Weather will vary by game, so just return typical
            weather = 'Typical conditions'
            return stadium, weather
        
        return 'Unknown', 'Unknown'
        
    def predict_current_week(self):
        """Predict current week games using pure hybrid model"""
        print_section_header("Pure Hybrid Model - Current Week Predictions")
        
        # Load model
        self.load_hybrid_model()
        
        # Get current week games
        current_week_games = self.get_current_week_games()
        
        print(f"📊 Predicting {len(current_week_games)} current week games")
        
        predictions = []
        
        for i, game in enumerate(current_week_games, 1):
            # Create feature vector for this specific game
            feature_vector = self.create_game_feature_vector(game)
            
            feature_array = np.array(feature_vector).reshape(1, -1)
            
            # Make predictions
            ml_prob = self.hybrid_ml_model.predict_proba(feature_array)[0]
            spread_prob = self.hybrid_spread_model.predict_proba(feature_array)[0]
            
            # Validate and cap extreme probabilities
            home_win_prob = self.validate_probability(ml_prob[1])
            away_win_prob = self.validate_probability(ml_prob[0])
            home_cover_prob = self.validate_probability(spread_prob[1])
            away_cover_prob = self.validate_probability(spread_prob[0])
            
            # Look up stadium and weather from historical data
            stadium, weather = self.lookup_stadium_weather(game['home_team'], game['away_team'])
            
            # Calculate spread edge and bet quality
            spread_edge, spread_bet_quality, recommended_bet = self.calculate_edge_and_quality(
                home_cover_prob, game.get('home_ml_odds'), game.get('away_ml_odds')
            )
            
            # Calculate ML edges
            home_ml_edge, away_ml_edge = self.calculate_ml_edges(
                home_win_prob, away_win_prob, game.get('home_ml_odds'), game.get('away_ml_odds')
            )
            
            # Store prediction
            prediction = {
                'game_id': game.get('id'),
                'season': game['season'],
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'commence_time': game.get('commence_time'),
                'spread': game['spread'],
                'home_ml_odds': game.get('home_ml_odds'),
                'away_ml_odds': game.get('away_ml_odds'),
                'over_under': game.get('over_under'),
                'week': game.get('week'),
                'home_win_prob': home_win_prob,
                'away_win_prob': away_win_prob,
                'home_cover_prob': home_cover_prob,
                'away_cover_prob': away_cover_prob,
                'stadium': stadium,
                'weather': weather,
                'recommended_bet': recommended_bet,
                'confidence_level': self.get_confidence_level(home_cover_prob),
                'spread_edge': spread_edge,
                'bet_quality': spread_bet_quality,
                'home_ml_edge': home_ml_edge,
                'away_ml_edge': away_ml_edge,
                'actual_home_win': game.get('home_team_wins', None),
                'actual_home_cover': game.get('ats_home_covers', None)
            }
            
            predictions.append(prediction)
            
        return predictions
        
    def print_week_7_predictions(self, predictions: List[Dict]):
        """Print formatted Week 7 predictions"""
        print_section_header("🏈 PURE HYBRID MODEL - WEEK 7 PREDICTIONS")
        
        total_ml_correct = 0
        total_spread_correct = 0
        total_games = 0
        
        for i, pred in enumerate(predictions, 1):
            home_team = pred['home_team']
            away_team = pred['away_team']
            spread = pred['spread']
            home_win_prob = pred['home_win_prob']
            away_win_prob = pred['away_win_prob']
            home_cover_prob = pred['home_cover_prob']
            away_cover_prob = pred['away_cover_prob']
            stadium = pred['stadium']
            weather = pred['weather']
            season = pred['season']
            home_ml_odds = pred.get('home_ml_odds')
            away_ml_odds = pred.get('away_ml_odds')
            home_ml_edge = pred.get('home_ml_edge')
            away_ml_edge = pred.get('away_ml_edge')
            
            # Get team records if available
            home_record = ""
            away_record = ""
            if self.feature_fetcher and self.feature_fetcher.team_features is not None:
                home_abbr = self.normalize_team_name_for_api(home_team)
                away_abbr = self.normalize_team_name_for_api(away_team)
                
                home_team_data = self.feature_fetcher.team_features[self.feature_fetcher.team_features['team'] == home_abbr]
                away_team_data = self.feature_fetcher.team_features[self.feature_fetcher.team_features['team'] == away_abbr]
                
                if len(home_team_data) > 0:
                    home_rec = home_team_data.iloc[0]
                    home_record = f" ({int(home_rec['wins'])}-{int(home_rec['losses'])})"
                
                if len(away_team_data) > 0:
                    away_rec = away_team_data.iloc[0]
                    away_record = f" ({int(away_rec['wins'])}-{int(away_rec['losses'])})"
            
            print(f"\n📍 GAME {i}: {away_team}{away_record} @ {home_team}{home_record} ({season})")
            print(f"   Spread: {spread}")
            print(f"   Stadium: {stadium}")
            print(f"   Weather: {weather}")
            
            # Show ML odds if available
            if home_ml_odds is not None and away_ml_odds is not None:
                home_odds_display = f"{home_ml_odds:+.0f}" if home_ml_odds < 0 else f"+{home_ml_odds:.0f}"
                away_odds_display = f"{away_ml_odds:+.0f}" if away_ml_odds < 0 else f"+{away_ml_odds:.0f}"
                print(f"   ML Odds: {home_team} ({home_odds_display}) | {away_team} ({away_odds_display})")
            
            print(f"\n🎯 MONEYLINE PREDICTIONS:")
            print(f"   {home_team} wins: {home_win_prob:.1%}")
            print(f"   {away_team} wins: {away_win_prob:.1%}")
            
            # Show ML edges if available
            if home_ml_edge is not None and away_ml_edge is not None:
                print(f"\n💰 MONEYLINE EDGES:")
                home_edge_display = f"{home_ml_edge:+.1f}%"
                away_edge_display = f"{away_ml_edge:+.1f}%"
                print(f"   {home_team} ML: {home_edge_display}")
                print(f"   {away_team} ML: {away_edge_display}")
                
                # Highlight best ML bet if edge > 3%
                if home_ml_edge > 3:
                    print(f"   ⭐ RECOMMENDED: {home_team} ML")
                elif away_ml_edge > 3:
                    print(f"   ⭐ RECOMMENDED: {away_team} ML")
            
            print(f"\n📊 SPREAD PREDICTIONS (Spread: {spread}):")
            print(f"   {home_team} covers ({spread:+.1f}): {home_cover_prob:.1%}")
            print(f"   {away_team} covers ({-spread:+.1f}): {away_cover_prob:.1%}")
            
            # Calculate edge
            implied_prob = 0.5  # 50/50 for spread
            edge = home_cover_prob - implied_prob
            edge_pct = edge * 100
            
            print(f"\n💰 SPREAD EDGE ANALYSIS:")
            print(f"   Our Probability: {home_cover_prob:.1%}")
            print(f"   Implied Probability: {implied_prob:.1%}")
            print(f"   Edge: {edge_pct:+.1f}%")
            
            if abs(edge_pct) > 10:
                quality = "EXCELLENT"
            elif abs(edge_pct) > 5:
                quality = "GOOD"
            elif abs(edge_pct) > 2:
                quality = "FAIR"
            else:
                quality = "NEUTRAL"
                
            print(f"   Quality: {quality}")
            
            # Show actual results if available
            if pred['actual_home_win'] is not None:
                actual_home_win = "✓" if pred['actual_home_win'] == 1 else "✗"
                actual_home_cover = "✓" if pred['actual_home_cover'] == 1 else "✗"
                print(f"\n📈 ACTUAL RESULTS:")
                print(f"   {home_team} won: {actual_home_win}")
                print(f"   {home_team} covered: {actual_home_cover}")
                
                # Check prediction accuracy
                ml_correct = "✓" if (home_win_prob > 0.5) == (pred['actual_home_win'] == 1) else "✗"
                spread_correct = "✓" if (home_cover_prob > 0.5) == (pred['actual_home_cover'] == 1) else "✗"
                print(f"   ML Prediction: {ml_correct}")
                print(f"   Spread Prediction: {spread_correct}")
                
                # Count accuracy
                if ml_correct == "✓":
                    total_ml_correct += 1
                if spread_correct == "✓":
                    total_spread_correct += 1
                total_games += 1
        
        # Print summary
        if total_games > 0:
            print(f"\n📊 WEEK 7 SUMMARY:")
            print(f"   ML Accuracy: {total_ml_correct}/{total_games} ({total_ml_correct/total_games:.1%})")
            print(f"   Spread Accuracy: {total_spread_correct}/{total_games} ({total_spread_correct/total_games:.1%})")
        
    def run_full_pipeline(self):
        """Run the complete pure hybrid pipeline"""
        print_section_header("🏈 PURE HYBRID MODEL PIPELINE")
        
        # Run dependencies
        self.run_dependencies()
        
        # Predict current week
        predictions = self.predict_current_week()
        
        # Print results
        self.print_week_7_predictions(predictions)
        
        # Store predictions in Supabase if available
        if self.supabase:
            self.store_predictions_in_supabase(predictions)
        
        print_section_header("🎉 PURE HYBRID PIPELINE COMPLETE!")
        print("✅ The hybrid model is THE model - combining historical + advanced features")
        print("✅ Week 7 predictions completed with weather/stadium + PBP/NGS data")

    def store_predictions_in_supabase(self, predictions: List[Dict]):
        """Store predictions in Supabase database"""
        try:
            print_section_header("Storing Predictions in Supabase")
            
            # Convert predictions to database format
            db_predictions = []
            for pred in predictions:
                db_pred = {
                    'week': int(pred.get('week', 7)),  # Current week
                    'season': int(pred.get('season', 2025)),
                    'game_date': pred.get('commence_time', '2025-10-19T00:00:00Z'),
                    'away_team': pred['away_team'],
                    'home_team': pred['home_team'],
                    'spread': float(pred['spread']) if pred['spread'] is not None else None,
                    'over_under': float(pred.get('over_under')) if pred.get('over_under') is not None else None,
                    'home_ml_odds': int(float(pred.get('home_ml_odds')) * 100) if pred.get('home_ml_odds') is not None else None,
                    'away_ml_odds': int(float(pred.get('away_ml_odds')) * 100) if pred.get('away_ml_odds') is not None else None,
                    'home_ml_prob': float(pred['home_win_prob']),
                    'away_ml_prob': float(pred['away_win_prob']),
                    'home_cover_prob': float(pred['home_cover_prob']),
                    'away_cover_prob': float(pred['away_cover_prob']),
                    'recommended_bet': pred.get('recommended_bet', ''),
                    'confidence_level': pred.get('confidence_level', ''),
                    'edge_percentage': float(pred.get('spread_edge', 0.0)),
                    'bet_quality': pred.get('bet_quality', ''),
                    'prediction_timestamp': datetime.now().isoformat(),
                    'model_version': 'pure_hybrid_v1',
                    'odds_source': 'the_odds_api'
                    # ML edges stored separately once DB columns are added:
                    # 'home_ml_edge': float(pred.get('home_ml_edge', 0.0)) if pred.get('home_ml_edge') is not None else None,
                    # 'away_ml_edge': float(pred.get('away_ml_edge', 0.0)) if pred.get('away_ml_edge') is not None else None,
                }
                db_predictions.append(db_pred)
            
            # Store in Supabase - upsert based on unique game combination
            # This will update existing predictions for the same week/season/teams
            result = self.supabase.table('nfl_model_predictions').upsert(
                db_predictions,
                on_conflict='week,season,away_team,home_team'
            ).execute()
            
            print(f"✅ Stored {len(db_predictions)} predictions in Supabase")
            print(f"📊 Database result: {len(result.data)} records")
            
        except Exception as e:
            print(f"❌ Error storing predictions in Supabase: {e}")
            logger.error(f"Supabase storage error: {e}")


if __name__ == "__main__":
    pipeline = PureHybridPipeline()
    pipeline.run_full_pipeline()
