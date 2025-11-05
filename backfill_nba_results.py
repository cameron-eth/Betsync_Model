#!/usr/bin/env python3
"""
NBA Results Backfill Script
Runs daily to backfill game results from previous day
Updates: actual scores, prediction correctness
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
SUPABASE_KEY = os.getenv('SUPABASE_KEY') or os.getenv('SUPABASE_ANON_KEY') or os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY')
ODDS_API_KEY = os.getenv('ODDS_API_KEY') or os.getenv('NEXT_PUBLIC_ODDS_API_KEY')
ODDS_API_BASE = 'https://api.the-odds-api.com/v4'

# NBA team name mappings (Odds API -> Your DB format)
TEAM_NAME_MAPPING = {
    'Atlanta Hawks': 'Atlanta Hawks',
    'Boston Celtics': 'Boston Celtics',
    'Brooklyn Nets': 'Brooklyn Nets',
    'Charlotte Hornets': 'Charlotte Hornets',
    'Chicago Bulls': 'Chicago Bulls',
    'Cleveland Cavaliers': 'Cleveland Cavaliers',
    'Dallas Mavericks': 'Dallas Mavericks',
    'Denver Nuggets': 'Denver Nuggets',
    'Detroit Pistons': 'Detroit Pistons',
    'Golden State Warriors': 'Golden State Warriors',
    'Houston Rockets': 'Houston Rockets',
    'Indiana Pacers': 'Indiana Pacers',
    'LA Clippers': 'Los Angeles Clippers',
    'Los Angeles Clippers': 'Los Angeles Clippers',
    'LA Lakers': 'Los Angeles Lakers',
    'Los Angeles Lakers': 'Los Angeles Lakers',
    'Memphis Grizzlies': 'Memphis Grizzlies',
    'Miami Heat': 'Miami Heat',
    'Milwaukee Bucks': 'Milwaukee Bucks',
    'Minnesota Timberwolves': 'Minnesota Timberwolves',
    'New Orleans Pelicans': 'New Orleans Pelicans',
    'New York Knicks': 'New York Knicks',
    'Oklahoma City Thunder': 'Oklahoma City Thunder',
    'Orlando Magic': 'Orlando Magic',
    'Philadelphia 76ers': 'Philadelphia 76ers',
    'Phoenix Suns': 'Phoenix Suns',
    'Portland Trail Blazers': 'Portland Trail Blazers',
    'Sacramento Kings': 'Sacramento Kings',
    'San Antonio Spurs': 'San Antonio Spurs',
    'Toronto Raptors': 'Toronto Raptors',
    'Utah Jazz': 'Utah Jazz',
    'Washington Wizards': 'Washington Wizards',
}


class NBAResultsBackfill:
    """Backfill NBA game results and update prediction accuracy"""
    
    def __init__(self):
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("Missing Supabase credentials")
        
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.odds_api_key = ODDS_API_KEY
        
    def normalize_team_name(self, team: str) -> str:
        """Normalize team names to match database format"""
        # Try direct mapping
        if team in TEAM_NAME_MAPPING:
            return TEAM_NAME_MAPPING[team]
        
        # Return original if no mapping found
        return team
    
    def get_games_needing_results(self, days_back: int = 3) -> List[Dict]:
        """
        Get all games from last few days that need results backfilled
        Default: last 3 days (NBA games happen daily)
        """
        print(f"\n🔍 Fetching games from last {days_back} days needing results...")
        
        cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        
        try:
            response = self.supabase.table('nba_predictions')\
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
    
    def fetch_scores_from_odds_api(self) -> List[Dict]:
        """
        Fetch scores from The Odds API for completed NBA games
        """
        if not self.odds_api_key:
            print("⚠️  No Odds API key - skipping score fetch")
            return []
        
        print(f"\n🏀 Fetching scores from Odds API...")
        
        try:
            # Get scores for NBA games
            url = f"{ODDS_API_BASE}/sports/basketball_nba/scores"
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
        Calculate if ML prediction was correct (NBA doesn't use spreads)
        """
        # Extract scores
        home_score = None
        away_score = None
        
        for team_score in score['scores']:
            if team_score['name'] == score['home_team']:
                home_score = int(team_score['score'])
            elif team_score['name'] == score['away_team']:
                away_score = int(team_score['score'])
        
        if home_score is None or away_score is None:
            raise ValueError(f"Could not extract scores from: {score}")
        
        # Determine actual winner
        actual_winner = 'home' if home_score > away_score else 'away'
        
        # ML prediction correctness
        predicted_winner = 'home' if game.get('home_win_probability', 0) > game.get('away_win_probability', 0) else 'away'
        ml_correct = (predicted_winner == actual_winner)
        
        return {
            'actual_home_score': home_score,
            'actual_away_score': away_score,
            'actual_winner': actual_winner,
            'prediction_correct': ml_correct,
            'game_completed': True,
            'result_synced_at': datetime.now().isoformat()
        }
    
    def update_game_results(self, game_id: int, results: Dict) -> bool:
        """
        Update a game with actual results
        """
        try:
            self.supabase.table('nba_predictions')\
                .update(results)\
                .eq('id', game_id)\
                .execute()
            
            return True
            
        except Exception as e:
            print(f"❌ Error updating game {game_id}: {e}")
            return False
    
    def run_backfill(self, days_back: int = 3):
        """
        Main backfill process
        """
        print("=" * 80)
        print("🏀 NBA RESULTS BACKFILL")
        print("=" * 80)
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔍 Looking back: {days_back} days")
        
        # Step 1: Get games needing results
        games = self.get_games_needing_results(days_back)
        
        if not games:
            print("\n✅ No games need backfilling!")
            return
        
        # Step 2: Fetch scores from API
        scores = self.fetch_scores_from_odds_api()
        
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
            
            try:
                # Calculate correctness
                results = self.calculate_prediction_correctness(game, score)
                
                # Update database
                if self.update_game_results(game['id'], results):
                    updated_count += 1
                    
                    ml_icon = "✅" if results['prediction_correct'] else "❌"
                    
                    print(f"{ml_icon} ML | {game_str} | "
                          f"Final: {results['actual_home_score']}-{results['actual_away_score']}")
            except Exception as e:
                print(f"❌ Error processing {game_str}: {e}")
        
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
        backfill = NBAResultsBackfill()
        
        # Default: backfill last 3 days (NBA plays daily)
        days_back = int(sys.argv[1]) if len(sys.argv) > 1 else 3
        
        backfill.run_backfill(days_back=days_back)
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

