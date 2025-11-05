#!/usr/bin/env python3
"""
NFL Results Backfill Script
Runs every Tuesday to backfill game results from previous week
Updates: actual scores, prediction correctness, spread correctness
"""

import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import requests
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# Configuration
SUPABASE_URL = os.getenv('SUPABASE_URL') or os.getenv('NEXT_PUBLIC_SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_ANON_KEY') or os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY')
ODDS_API_KEY = os.getenv('ODDS_API_KEY') or os.getenv('NEXT_PUBLIC_ODDS_API_KEY')
ODDS_API_BASE = 'https://api.the-odds-api.com/v4'

# NFL team name mappings (Odds API -> Your DB)
TEAM_NAME_MAPPING = {
    'Arizona Cardinals': 'ARI',
    'Atlanta Falcons': 'ATL',
    'Baltimore Ravens': 'BAL',
    'Buffalo Bills': 'BUF',
    'Carolina Panthers': 'CAR',
    'Chicago Bears': 'CHI',
    'Cincinnati Bengals': 'CIN',
    'Cleveland Browns': 'CLE',
    'Dallas Cowboys': 'DAL',
    'Denver Broncos': 'DEN',
    'Detroit Lions': 'DET',
    'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU',
    'Indianapolis Colts': 'IND',
    'Jacksonville Jaguars': 'JAX',
    'Kansas City Chiefs': 'KC',
    'Las Vegas Raiders': 'LV',
    'Los Angeles Chargers': 'LAC',
    'Los Angeles Rams': 'LAR',
    'Miami Dolphins': 'MIA',
    'Minnesota Vikings': 'MIN',
    'New England Patriots': 'NE',
    'New Orleans Saints': 'NO',
    'New York Giants': 'NYG',
    'New York Jets': 'NYJ',
    'Philadelphia Eagles': 'PHI',
    'Pittsburgh Steelers': 'PIT',
    'San Francisco 49ers': 'SF',
    'Seattle Seahawks': 'SEA',
    'Tampa Bay Buccaneers': 'TB',
    'Tennessee Titans': 'TEN',
    'Washington Commanders': 'WAS',
}


class NFLResultsBackfill:
    """Backfill NFL game results and update prediction accuracy"""
    
    def __init__(self):
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("Missing Supabase credentials")
        
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.odds_api_key = ODDS_API_KEY
        
    def normalize_team_name(self, team: str) -> str:
        """Normalize team names to abbreviations"""
        # If already abbreviated, return as-is
        if len(team) <= 3:
            return team
        
        # Try direct mapping
        if team in TEAM_NAME_MAPPING:
            return TEAM_NAME_MAPPING[team]
        
        # Return original if no mapping found (will try fuzzy match later)
        return team
    
    def get_games_needing_results(self, days_back: int = 7) -> List[Dict]:
        """
        Get all games from last week that need results backfilled
        Default: last 7 days (covers previous week's games)
        """
        print(f"\n🔍 Fetching games from last {days_back} days needing results...")
        
        cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        
        try:
            response = self.supabase.table('nfl_model_predictions')\
                .select('*')\
                .eq('game_completed', False)\
                .gte('game_date', cutoff_date)\
                .lt('game_date', datetime.now().isoformat())\
                .execute()
            
            games = response.data or []
            print(f"✅ Found {len(games)} games needing results")
            return games
            
        except Exception as e:
            print(f"❌ Error fetching games: {e}")
            return []
    
    def fetch_scores_from_odds_api(self, game_date_start: str, game_date_end: str) -> List[Dict]:
        """
        Fetch scores from The Odds API for completed games
        """
        if not self.odds_api_key:
            print("⚠️  No Odds API key - skipping score fetch")
            return []
        
        print(f"\n📊 Fetching scores from Odds API...")
        
        try:
            # Get scores for NFL games (endpoint doesn't use trailing slash)
            url = f"{ODDS_API_BASE}/sports/americanfootball_nfl/scores"
            params = {
                'apiKey': self.odds_api_key,
                'daysFrom': 3  # Look back 3 days for completed games
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            scores = response.json()
            print(f"✅ Fetched {len(scores)} games from Odds API")
            
            return scores
            
        except requests.exceptions.HTTPError as e:
            print(f"❌ Error fetching scores (HTTP {e.response.status_code}): {e}")
            print(f"   Response: {e.response.text[:200]}")
            return []
        except Exception as e:
            print(f"❌ Error fetching scores: {e}")
            return []
    
    def match_game_to_score(self, game: Dict, scores: List[Dict]) -> Optional[Dict]:
        """
        Match a prediction to its actual score
        """
        game_date = game['game_date'][:10]  # YYYY-MM-DD
        home_team = self.normalize_team_name(game['home_team'])
        away_team = self.normalize_team_name(game['away_team'])
        
        for score in scores:
            if not score.get('completed'):
                continue
            
            score_date = score['commence_time'][:10]
            score_home = self.normalize_team_name(score['home_team'])
            score_away = self.normalize_team_name(score['away_team'])
            
            # Match by date and teams
            if (score_date == game_date and 
                score_home == home_team and 
                score_away == away_team):
                return score
        
        return None
    
    def calculate_prediction_correctness(self, game: Dict, score: Dict) -> Dict:
        """
        Calculate if ML and spread predictions were correct
        """
        home_score = score['scores'][0]['score'] if score['scores'][0]['name'] == score['home_team'] else score['scores'][1]['score']
        away_score = score['scores'][1]['score'] if score['scores'][1]['name'] == score['away_team'] else score['scores'][0]['score']
        
        # Convert to int
        home_score = int(home_score)
        away_score = int(away_score)
        
        # Determine actual winner
        actual_winner = 'home' if home_score > away_score else 'away'
        
        # ML prediction correctness
        predicted_winner = 'home' if game.get('home_ml_prob', 0) > game.get('away_ml_prob', 0) else 'away'
        ml_correct = (predicted_winner == actual_winner)
        
        # Spread prediction correctness
        spread = game.get('spread', 0) or 0
        home_covered = (home_score + spread) > away_score
        away_covered = not home_covered
        
        # Check if we predicted spread correctly
        predicted_spread_winner = 'home' if game.get('home_cover_prob', 0) > game.get('away_cover_prob', 0) else 'away'
        
        # Determine actual spread result
        margin = home_score - away_score
        if abs(margin + spread) < 0.5:  # Push
            actual_spread_covered = 'push'
            spread_correct = False  # Count pushes as incorrect for accuracy
        else:
            actual_spread_covered = 'home' if home_covered else 'away'
            spread_correct = (predicted_spread_winner == actual_spread_covered)
        
        return {
            'actual_home_score': home_score,
            'actual_away_score': away_score,
            'actual_winner': actual_winner,
            'actual_spread_covered': actual_spread_covered,
            'prediction_correct': ml_correct,
            'spread_correct': spread_correct,
            'game_completed': True,
            'result_synced_at': datetime.now().isoformat()
        }
    
    def update_game_results(self, game_id: int, results: Dict) -> bool:
        """
        Update a game with actual results
        """
        try:
            self.supabase.table('nfl_model_predictions')\
                .update(results)\
                .eq('id', game_id)\
                .execute()
            
            return True
            
        except Exception as e:
            print(f"❌ Error updating game {game_id}: {e}")
            return False
    
    def run_backfill(self, days_back: int = 7):
        """
        Main backfill process
        """
        print("=" * 80)
        print("🏈 NFL RESULTS BACKFILL - TUESDAY EDITION")
        print("=" * 80)
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔍 Looking back: {days_back} days")
        
        # Step 1: Get games needing results
        games = self.get_games_needing_results(days_back)
        
        if not games:
            print("\n✅ No games need backfilling!")
            return
        
        # Step 2: Fetch scores from API
        game_dates = [g['game_date'] for g in games]
        earliest_date = min(game_dates)
        latest_date = max(game_dates)
        
        scores = self.fetch_scores_from_odds_api(earliest_date, latest_date)
        
        if not scores:
            print("⚠️  No scores available from API")
            return
        
        # Step 3: Match and update
        print(f"\n🔄 Processing {len(games)} games...")
        
        updated_count = 0
        matched_count = 0
        
        for game in games:
            game_str = f"{game['away_team']} @ {game['home_team']} ({game['game_date'][:10]})"
            
            # Find matching score
            score = self.match_game_to_score(game, scores)
            
            if not score:
                print(f"⚠️  No score found for: {game_str}")
                continue
            
            matched_count += 1
            
            # Calculate correctness
            results = self.calculate_prediction_correctness(game, score)
            
            # Update database
            if self.update_game_results(game['id'], results):
                updated_count += 1
                
                ml_icon = "✅" if results['prediction_correct'] else "❌"
                spread_icon = "✅" if results['spread_correct'] else "❌"
                
                print(f"{ml_icon} ML {spread_icon} Spread | {game_str} | "
                      f"Final: {results['actual_home_score']}-{results['actual_away_score']}")
        
        # Summary
        print("\n" + "=" * 80)
        print("📊 BACKFILL SUMMARY")
        print("=" * 80)
        print(f"Games found: {len(games)}")
        print(f"Scores matched: {matched_count}")
        print(f"Database updated: {updated_count}")
        
        print("=" * 80)
        print("✅ Backfill complete!")


def main():
    """Entry point"""
    try:
        backfill = NFLResultsBackfill()
        
        # Default: backfill last 7 days (run every Tuesday for previous week)
        days_back = int(sys.argv[1]) if len(sys.argv) > 1 else 7
        
        backfill.run_backfill(days_back=days_back)
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

