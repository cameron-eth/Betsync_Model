"""
Update actual game results for completed games to track model performance.
This script should be run after games are completed to update the database
with actual winners and scores.
"""

import os
import sys
from datetime import datetime, timedelta
from supabase import create_client
from dotenv import load_dotenv

# Load environment
load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_KEY')  # Use service key for admin operations

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Error: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
    sys.exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def update_nfl_results():
    """Update NFL game results from ESPN or other source"""
    print("🏈 Updating NFL game results...")
    
    # Get predictions for games that should be completed
    # (games from yesterday and earlier that aren't marked as completed)
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    response = supabase.table('nfl_model_predictions')\
        .select('*')\
        .eq('game_completed', False)\
        .lte('game_date', yesterday)\
        .execute()
    
    pending_games = response.data
    print(f"Found {len(pending_games)} pending NFL games to update")
    
    # TODO: Integrate with ESPN API or NFL data source to get actual scores
    # For now, this is a manual update placeholder
    
    # Example update (you'll replace this with actual API calls):
    # for game in pending_games:
    #     # Fetch actual result from API
    #     actual_home_score, actual_away_score = fetch_nfl_score(game['home_team'], game['away_team'], game['game_date'])
    #     actual_winner = game['home_team'] if actual_home_score > actual_away_score else game['away_team']
    #     
    #     # Update database
    #     supabase.table('nfl_model_predictions').update({
    #         'actual_winner': actual_winner,
    #         'home_score': actual_home_score,
    #         'away_score': actual_away_score,
    #         'game_completed': True,
    #         'completed_at': datetime.now().isoformat()
    #     }).eq('id', game['id']).execute()
    
    return len(pending_games)


def update_nba_results():
    """Update NBA game results from NBA API"""
    print("🏀 Updating NBA game results...")
    
    try:
        from nba_api.stats.endpoints import scoreboardv2
        from nba_api.stats.static import teams as nba_teams
        import time
        
        # Get predictions for games that should be completed
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        today = datetime.now().strftime('%Y-%m-%d')
        
        response = supabase.table('nba_predictions')\
            .select('*')\
            .eq('game_completed', False)\
            .gte('game_date', yesterday)\
            .lte('game_date', today)\
            .execute()
        
        pending_games = response.data
        print(f"Found {len(pending_games)} pending NBA games to update")
        
        # Fetch scoreboard for the date range
        all_teams = nba_teams.get_teams()
        updated_count = 0
        
        for game in pending_games:
            try:
                game_date = game['game_date']
                
                # Query NBA API for scores on this date
                scoreboard = scoreboardv2.ScoreboardV2(game_date=game_date)
                time.sleep(0.6)  # Rate limiting
                
                games_data = scoreboard.get_data_frames()[0]
                
                # Find matching game
                home_team_name = game['home_team']
                away_team_name = game['away_team']
                
                for _, game_row in games_data.iterrows():
                    home_team_id = game_row['HOME_TEAM_ID']
                    visitor_team_id = game_row['VISITOR_TEAM_ID']
                    
                    # Get team names from IDs
                    home_team_info = next((t for t in all_teams if t['id'] == home_team_id), None)
                    away_team_info = next((t for t in all_teams if t['id'] == visitor_team_id), None)
                    
                    if home_team_info and away_team_info:
                        if home_team_info['full_name'] == home_team_name and away_team_info['full_name'] == away_team_name:
                            # Found the game!
                            home_score = int(game_row['PTS_home']) if 'PTS_home' in game_row else None
                            away_score = int(game_row['PTS_away']) if 'PTS_away' in game_row else None
                            
                            if home_score is not None and away_score is not None:
                                actual_winner = home_team_name if home_score > away_score else away_team_name
                                
                                # Update database
                                supabase.table('nba_predictions').update({
                                    'actual_winner': actual_winner,
                                    'home_score': home_score,
                                    'away_score': away_score,
                                    'game_completed': True,
                                    'completed_at': datetime.now().isoformat()
                                }).eq('id', game['id']).execute()
                                
                                print(f"✅ Updated: {away_team_name} @ {home_team_name} - {actual_winner} won ({away_score}-{home_score})")
                                updated_count += 1
                                break
                
            except Exception as e:
                print(f"Error updating game {game['id']}: {e}")
                continue
        
        print(f"✅ Updated {updated_count} NBA games")
        return updated_count
        
    except ImportError:
        print("⚠️  nba_api not installed. Install with: pip install nba_api")
        return 0
    except Exception as e:
        print(f"Error updating NBA results: {e}")
        return 0


def main():
    """Main function to update game results"""
    print("\n" + "="*50)
    print("📊 Updating Game Results for Performance Tracking")
    print("="*50 + "\n")
    
    nfl_count = update_nfl_results()
    nba_count = update_nba_results()
    
    print("\n" + "="*50)
    print(f"✅ Update Complete!")
    print(f"   NFL: {nfl_count} games processed")
    print(f"   NBA: {nba_count} games updated")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()

