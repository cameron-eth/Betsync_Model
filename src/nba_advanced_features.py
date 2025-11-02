"""
NBA Advanced Feature Engineering using nba_api

Extracts rich features from:
- Box Scores (traditional stats)
- Team Stats (offensive/defensive ratings, pace, efficiency)
- Player Tracking Stats (speed, distance, touches)
- Team Game Logs (rolling averages, recent form)
- Injury Data (player availability, impact scores)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import time

# Setup logger first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from nba_api.stats.endpoints import (
        teamgamelog, leaguegamelog, boxscoretraditionalv2,
        leaguedashteamstats, boxscoreplayertrackv2, scheduleleaguev2,
        commonteamyears
    )
    from nba_api.live.nba.endpoints import scoreboard
    NBA_API_AVAILABLE = True
except ImportError as e:
    NBA_API_AVAILABLE = False
    logger.warning(f"nba_api not available: {e}")

try:
    from .nba_config import (
        NBA_SEASONS_AVAILABLE, NBA_TEAMS, NBA_TEAMS_REVERSE,
        LAST_N_GAMES, MIN_GAMES_PLAYED, BACK_TO_BACK_REST_DAYS
    )
    from .nba_utils import (
        logger, print_section_header, normalize_team_name,
        calculate_offensive_rating, calculate_defensive_rating,
        calculate_net_rating, calculate_true_shooting_percentage,
        calculate_effective_fg_percentage, calculate_pace,
        safe_float, safe_int, get_season_from_date
    )
    from .nba_injury_features import NBAInjuryFeatureEngine
except ImportError:
    from nba_config import (
        NBA_SEASONS_AVAILABLE, NBA_TEAMS, NBA_TEAMS_REVERSE,
        LAST_N_GAMES, MIN_GAMES_PLAYED, BACK_TO_BACK_REST_DAYS
    )
    from nba_utils import (
        logger, print_section_header, normalize_team_name, get_team_abbr,
        calculate_offensive_rating, calculate_defensive_rating,
        calculate_net_rating, calculate_true_shooting_percentage,
        calculate_effective_fg_percentage, calculate_pace,
        safe_float, safe_int, get_season_from_date
    )
    from nba_injury_features import NBAInjuryFeatureEngine


class NBAAdvancedFeatureEngine:
    """Extract advanced features from nba_api data"""
    
    def __init__(self, seasons: List[int] = None, include_injuries: bool = True):
        """Initialize with seasons to load"""
        if seasons is None:
            seasons = NBA_SEASONS_AVAILABLE
        self.seasons = seasons
        self.include_injuries = include_injuries
        
        # Data storage
        self.schedules_df = None
        self.game_logs_df = None
        self.team_stats_df = None
        self.player_tracking_df = None
        
        # Team ID mappings
        self.team_id_map = {}  # team_id -> abbreviation
        self.team_abbr_map = {}  # abbreviation -> team_id
        
        # Injury feature engine
        self.injury_engine = None
        if self.include_injuries:
            try:
                self.injury_engine = NBAInjuryFeatureEngine(seasons=seasons)
            except Exception as e:
                logger.warning(f"Could not initialize injury engine: {e}")
                self.injury_engine = None
        
        # Load team mappings
        self._load_team_mappings()
    
    def _load_team_mappings(self):
        """Load NBA team ID mappings from nba_api"""
        if not NBA_API_AVAILABLE:
            logger.warning("nba_api not available, skipping team mappings")
            return
            
        try:
            # Use commonteamyears to get team info
            teams_df = commonteamyears.CommonTeamYears().get_data_frames()[0]
            
            # Create mappings from the dataframe
            for _, row in teams_df.iterrows():
                team_id = row.get('TEAM_ID')
                abbr = row.get('ABBREVIATION')
                
                if team_id and abbr:
                    self.team_id_map[team_id] = abbr
                    self.team_abbr_map[abbr] = team_id
                
        except Exception as e:
            logger.warning(f"Could not load team mappings: {e}")
    
    def get_team_id(self, team_abbr: str) -> Optional[int]:
        """Get team ID from abbreviation"""
        return self.team_abbr_map.get(team_abbr)
    
    def get_team_abbr(self, team_id: int) -> Optional[str]:
        """Get team abbreviation from ID"""
        return self.team_id_map.get(team_id)
    
    def load_all_data(self):
        """Load all data sources from nba_api"""
        print_section_header("Loading Advanced NBA Data")
        
        print(f"📊 Loading data for seasons: {self.seasons}")
        
        # Load schedules for all seasons
        print("\n1️⃣ Loading Schedules...")
        try:
            self._load_schedules()
            print(f"   ✅ Loaded {len(self.schedules_df)} games")
        except Exception as e:
            logger.error(f"Error loading schedules: {e}")
            print(f"   ⚠️ Error loading schedules: {e}")
        
        # Load team game logs
        print("\n2️⃣ Loading Team Game Logs...")
        try:
            self._load_game_logs()
            print(f"   ✅ Loaded {len(self.game_logs_df)} game records")
        except Exception as e:
            logger.error(f"Error loading game logs: {e}")
            print(f"   ⚠️ Error loading game logs: {e}")
        
        # Load team stats (league dashboard)
        print("\n3️⃣ Loading Team Stats...")
        try:
            self._load_team_stats()
            print(f"   ✅ Loaded team stats")
        except Exception as e:
            logger.error(f"Error loading team stats: {e}")
            print(f"   ⚠️ Error loading team stats: {e}")
        
        # Injury Data
        if self.include_injuries and self.injury_engine:
            print("\n4️⃣ Loading Injury Data...")
            try:
                self.injury_engine.load_all_data()
                print(f"   ✅ Injury data loaded successfully")
            except Exception as e:
                print(f"   ⚠️ Injury data not available: {e}")
                self.injury_engine = None
        
        print("\n✅ All data loaded successfully!\n")
    
    def _load_schedules(self):
        """Load game schedules for all seasons"""
        if not NBA_API_AVAILABLE:
            logger.error("nba_api not available, cannot load schedules")
            self.schedules_df = pd.DataFrame()
            return
            
        all_schedules = []
        
        for season in self.seasons:
            try:
                # Format: "2023-24" for 2023-24 season
                season_str = f"{season-1}-{str(season)[-2:]}"
                
                schedule = scheduleleaguev2.ScheduleLeagueV2(
                    season=season_str,
                    league_id='00'
                )
                df = schedule.get_data_frames()[0]
                
                if len(df) > 0:
                    df['season'] = season
                    all_schedules.append(df)
                
                # Rate limiting
                time.sleep(0.6)
                
            except Exception as e:
                logger.warning(f"Could not load schedule for {season}: {e}")
                continue
        
        if all_schedules:
            self.schedules_df = pd.concat(all_schedules, ignore_index=True)
        else:
            self.schedules_df = pd.DataFrame()
    
    def _load_game_logs(self):
        """Load team game logs for all seasons"""
        if not NBA_API_AVAILABLE:
            logger.error("nba_api not available, cannot load game logs")
            self.game_logs_df = pd.DataFrame()
            return
            
        all_logs = []
        
        for season in self.seasons:
            try:
                season_str = f"{season-1}-{str(season)[-2:]}"
                
                # Use league game log to get all teams
                league_log = leaguegamelog.LeagueGameLog(
                    season=season_str,
                    season_type_all_star='Regular Season'
                )
                df = league_log.get_data_frames()[0]
                
                if len(df) > 0:
                    df['season'] = season
                    all_logs.append(df)
                
                time.sleep(0.6)
                
            except Exception as e:
                logger.warning(f"Could not load game logs for {season}: {e}")
                continue
        
        if all_logs:
            self.game_logs_df = pd.concat(all_logs, ignore_index=True)
        else:
            self.game_logs_df = pd.DataFrame()
    
    def _load_team_stats(self):
        """Load league-wide team stats"""
        if not NBA_API_AVAILABLE:
            logger.error("nba_api not available, cannot load team stats")
            self.team_stats_df = pd.DataFrame()
            return
            
        all_stats = []
        
        for season in self.seasons:
            try:
                season_str = f"{season-1}-{str(season)[-2:]}"
                
                team_stats = leaguedashteamstats.LeagueDashTeamStats(
                    season=season_str,
                    season_type_all_star='Regular Season',
                    per_mode_simple='PerGame'
                )
                df = team_stats.get_data_frames()[0]
                
                if len(df) > 0:
                    df['season'] = season
                    all_stats.append(df)
                
                time.sleep(0.6)
                
            except Exception as e:
                logger.warning(f"Could not load team stats for {season}: {e}")
                continue
        
        if all_stats:
            self.team_stats_df = pd.concat(all_stats, ignore_index=True)
        else:
            self.team_stats_df = pd.DataFrame()
    
    def calculate_team_features(self, team_abbr: str, season: int, game_date: str = None) -> Dict:
        """
        Calculate advanced features for a team
        
        Features:
        - Pace (possessions per 48 minutes)
        - Offensive/Defensive Rating
        - True Shooting %, eFG%
        - 3PT shooting (attempts, makes, %)
        - Free throw rate
        - Rebounding (offensive/defensive)
        - Turnover rate
        - Recent form (last N games)
        - Home/away splits
        - Rest days
        """
        
        features = {}
        
        # Filter game logs for this team
        if 'TEAM_ABBREVIATION' in self.game_logs_df.columns:
            team_col = 'TEAM_ABBREVIATION'
        elif 'TEAM_NAME' in self.game_logs_df.columns:
            team_col = 'TEAM_NAME'
        else:
            return self._get_default_team_features()
        
        team_logs = self.game_logs_df[
            (self.game_logs_df['season'] == season) &
            (self.game_logs_df[team_col] == team_abbr)
        ].copy()
        
        if len(team_logs) == 0:
            return self._get_default_team_features()
        
        # Sort by game date
        if 'GAME_DATE' in team_logs.columns:
            team_logs['GAME_DATE'] = pd.to_datetime(team_logs['GAME_DATE'])
            team_logs = team_logs.sort_values('GAME_DATE')
            
            # Filter out future games if game_date provided
            if game_date:
                game_dt = pd.to_datetime(game_date)
                team_logs = team_logs[team_logs['GAME_DATE'] < game_dt]
        
        if len(team_logs) == 0:
            return self._get_default_team_features()
        
        # Calculate season averages
        features.update(self._calculate_season_averages(team_logs))
        
        # Calculate last N games
        if len(team_logs) >= MIN_GAMES_PLAYED:
            last_n = team_logs.tail(LAST_N_GAMES)
            features.update(self._calculate_last_n_games(last_n, prefix='last5'))
        
        # Calculate home/away splits
        if 'MATCHUP' in team_logs.columns:
            home_games = team_logs[team_logs['MATCHUP'].str.contains('vs.', na=False)]
            away_games = team_logs[team_logs['MATCHUP'].str.contains('@', na=False)]
            
            if len(home_games) > 0:
                features.update(self._calculate_season_averages(home_games, prefix='home'))
            if len(away_games) > 0:
                features.update(self._calculate_season_averages(away_games, prefix='away'))
        
        # Calculate rest days for last game
        if game_date and 'GAME_DATE' in team_logs.columns and len(team_logs) > 0:
            last_game_date = team_logs['GAME_DATE'].iloc[-1]
            game_dt = pd.to_datetime(game_date)
            rest_days = (game_dt - last_game_date).days
            features['rest_days'] = max(0, rest_days)
            features['is_back_to_back'] = 1 if rest_days == BACK_TO_BACK_REST_DAYS else 0
        else:
            features['rest_days'] = 2  # Default
            features['is_back_to_back'] = 0
        
        return features
    
    def _calculate_season_averages(self, team_logs: pd.DataFrame, prefix: str = '') -> Dict:
        """Calculate season average features from game logs"""
        features = {}
        
        if len(team_logs) == 0:
            return self._get_default_team_features()
        
        # Points
        features[f'{prefix}_ppg'] = safe_float(team_logs['PTS'].mean()) if 'PTS' in team_logs.columns else 0.0
        features[f'{prefix}_points_allowed'] = safe_float(team_logs['PTS'].mean()) if 'PTS' in team_logs.columns else 0.0
        
        # Field Goals
        if 'FGM' in team_logs.columns and 'FGA' in team_logs.columns:
            features[f'{prefix}_fgm'] = safe_float(team_logs['FGM'].mean())
            features[f'{prefix}_fga'] = safe_float(team_logs['FGA'].mean())
            features[f'{prefix}_fg_pct'] = safe_float(team_logs['FG_PCT'].mean()) if 'FG_PCT' in team_logs.columns else 0.0
        else:
            features[f'{prefix}_fgm'] = 0.0
            features[f'{prefix}_fga'] = 0.0
            features[f'{prefix}_fg_pct'] = 0.0
        
        # 3-Pointers
        if 'FG3M' in team_logs.columns and 'FG3A' in team_logs.columns:
            features[f'{prefix}_fg3m'] = safe_float(team_logs['FG3M'].mean())
            features[f'{prefix}_fg3a'] = safe_float(team_logs['FG3A'].mean())
            features[f'{prefix}_fg3_pct'] = safe_float(team_logs['FG3_PCT'].mean()) if 'FG3_PCT' in team_logs.columns else 0.0
        else:
            features[f'{prefix}_fg3m'] = 0.0
            features[f'{prefix}_fg3a'] = 0.0
            features[f'{prefix}_fg3_pct'] = 0.0
        
        # Free Throws
        if 'FTM' in team_logs.columns and 'FTA' in team_logs.columns:
            features[f'{prefix}_ftm'] = safe_float(team_logs['FTM'].mean())
            features[f'{prefix}_fta'] = safe_float(team_logs['FTA'].mean())
            features[f'{prefix}_ft_pct'] = safe_float(team_logs['FT_PCT'].mean()) if 'FT_PCT' in team_logs.columns else 0.0
            features[f'{prefix}_ft_rate'] = safe_float(team_logs['FTA'].sum() / team_logs['FGA'].sum()) if features[f'{prefix}_fga'] > 0 else 0.0
        else:
            features[f'{prefix}_ftm'] = 0.0
            features[f'{prefix}_fta'] = 0.0
            features[f'{prefix}_ft_pct'] = 0.0
            features[f'{prefix}_ft_rate'] = 0.0
        
        # Rebounds
        if 'OREB' in team_logs.columns and 'DREB' in team_logs.columns:
            features[f'{prefix}_oreb'] = safe_float(team_logs['OREB'].mean())
            features[f'{prefix}_dreb'] = safe_float(team_logs['DREB'].mean())
            features[f'{prefix}_reb'] = safe_float(team_logs['REB'].mean()) if 'REB' in team_logs.columns else 0.0
        else:
            features[f'{prefix}_oreb'] = 0.0
            features[f'{prefix}_dreb'] = 0.0
            features[f'{prefix}_reb'] = 0.0
        
        # Assists, Steals, Blocks, Turnovers
        features[f'{prefix}_ast'] = safe_float(team_logs['AST'].mean()) if 'AST' in team_logs.columns else 0.0
        features[f'{prefix}_stl'] = safe_float(team_logs['STL'].mean()) if 'STL' in team_logs.columns else 0.0
        features[f'{prefix}_blk'] = safe_float(team_logs['BLK'].mean()) if 'BLK' in team_logs.columns else 0.0
        features[f'{prefix}_tov'] = safe_float(team_logs['TOV'].mean()) if 'TOV' in team_logs.columns else 0.0
        
        # Advanced metrics
        points = features[f'{prefix}_ppg'] * len(team_logs)
        fga = features[f'{prefix}_fga'] * len(team_logs)
        fta = features[f'{prefix}_fta'] * len(team_logs)
        fgm = features[f'{prefix}_fgm'] * len(team_logs)
        fg3m = features[f'{prefix}_fg3m'] * len(team_logs)
        
        features[f'{prefix}_ts_pct'] = calculate_true_shooting_percentage(points, fga, fta)
        features[f'{prefix}_efg_pct'] = calculate_effective_fg_percentage(fgm, fg3m, fga)
        
        # Turnover rate (per 100 possessions estimate)
        if features[f'{prefix}_fga'] > 0:
            # Estimate possessions: FGA + 0.44 * FTA + TOV - OREB
            estimated_poss = features[f'{prefix}_fga'] + 0.44 * features[f'{prefix}_fta'] + features[f'{prefix}_tov'] - features[f'{prefix}_oreb']
            if estimated_poss > 0:
                features[f'{prefix}_tov_rate'] = features[f'{prefix}_tov'] / estimated_poss
            else:
                features[f'{prefix}_tov_rate'] = 0.15
        else:
            features[f'{prefix}_tov_rate'] = 0.15
        
        # Wins/Losses
        if 'WL' in team_logs.columns:
            wins = (team_logs['WL'] == 'W').sum()
            losses = (team_logs['WL'] == 'L').sum()
            features[f'{prefix}_wins'] = wins
            features[f'{prefix}_losses'] = losses
            features[f'{prefix}_win_pct'] = wins / (wins + losses) if (wins + losses) > 0 else 0.5
        else:
            features[f'{prefix}_wins'] = 0
            features[f'{prefix}_losses'] = 0
            features[f'{prefix}_win_pct'] = 0.5
        
        return features
    
    def _calculate_last_n_games(self, team_logs: pd.DataFrame, prefix: str = 'last5') -> Dict:
        """Calculate last N games features"""
        return self._calculate_season_averages(team_logs, prefix=prefix)
    
    def _get_default_team_features(self) -> Dict:
        """Default team features when no data available"""
        return {
            'ppg': 100.0,
            'points_allowed': 100.0,
            'fgm': 38.0,
            'fga': 85.0,
            'fg_pct': 0.45,
            'fg3m': 12.0,
            'fg3a': 35.0,
            'fg3_pct': 0.35,
            'ftm': 12.0,
            'fta': 15.0,
            'ft_pct': 0.78,
            'ft_rate': 0.18,
            'oreb': 10.0,
            'dreb': 32.0,
            'reb': 42.0,
            'ast': 24.0,
            'stl': 7.0,
            'blk': 5.0,
            'tov': 14.0,
            'ts_pct': 0.55,
            'efg_pct': 0.52,
            'tov_rate': 0.15,
            'wins': 0,
            'losses': 0,
            'win_pct': 0.5,
            'rest_days': 2,
            'is_back_to_back': 0
        }
    
    def create_enhanced_dataset(self, output_path: str = None):
        """
        Create enhanced dataset with all features
        
        This combines:
        - Historical game results
        - Team features (pace, efficiency, shooting)
        - Last N games stats
        - Home/away splits
        - Injury features
        - Market odds (if available)
        """
        print_section_header("Creating Enhanced NBA Dataset")
        
        if output_path is None:
            from .nba_config import NBA_ENHANCED_FEATURES_CSV
            output_path = NBA_ENHANCED_FEATURES_CSV
        
        # Load historical matchups if available
        from .nba_config import NBA_HISTORIC_MATCHUPS_CSV
        if NBA_HISTORIC_MATCHUPS_CSV.exists():
            print("📊 Loading historical matchups...")
            matchups_df = pd.read_csv(NBA_HISTORIC_MATCHUPS_CSV)
            print(f"   ✅ Loaded {len(matchups_df)} historical games")
        else:
            print("⚠️  No historical matchups CSV found. Creating from schedules...")
            matchups_df = self._create_matchups_from_schedules()
        
        if len(matchups_df) == 0:
            raise ValueError("No matchup data available!")
        
        print(f"\n🔧 Engineering features for {len(matchups_df)} games...")
        
        enhanced_features = []
        
        for idx, row in matchups_df.iterrows():
            if idx % 100 == 0:
                print(f"   Processing game {idx+1}/{len(matchups_df)}...")
            
            try:
                features = self._create_game_features(row)
                enhanced_features.append(features)
            except Exception as e:
                logger.error(f"Error processing game {idx}: {e}")
                continue
        
        enhanced_df = pd.DataFrame(enhanced_features)
        
        print(f"\n✅ Created enhanced dataset with {len(enhanced_df)} games and {len(enhanced_df.columns)} features")
        print(f"   Saving to: {output_path}")
        
        enhanced_df.to_csv(output_path, index=False)
        print("✅ Enhanced dataset saved!")
        
        return enhanced_df
    
    def _create_matchups_from_schedules(self) -> pd.DataFrame:
        """Create matchups dataframe from schedules"""
        if self.schedules_df is None or len(self.schedules_df) == 0:
            return pd.DataFrame()
        
        matchups = []
        
        for _, row in self.schedules_df.iterrows():
            matchup = {
                'game_id': row.get('GAME_ID', ''),
                'game_date': row.get('GAME_DATE', ''),
                'home_team': row.get('HOME_TEAM_NAME', ''),
                'away_team': row.get('VISITOR_TEAM_NAME', ''),
                'season': row.get('season', 2024),
                'home_score': row.get('HOME_TEAM_SCORE', 0),
                'away_score': row.get('VISITOR_TEAM_SCORE', 0),
            }
            matchups.append(matchup)
        
        return pd.DataFrame(matchups)
    
    def _create_game_features(self, game_row: pd.Series) -> Dict:
        """Create features for a single game"""
        features = {}
        
        # Game info
        features['game_id'] = game_row.get('game_id', '')
        features['game_date'] = game_row.get('game_date', '')
        features['season'] = game_row.get('season', 2024)
        features['home_team'] = normalize_team_name(game_row.get('home_team', ''))
        features['away_team'] = normalize_team_name(game_row.get('away_team', ''))
        
        # Get team abbreviations
        home_abbr = get_team_abbr(features['home_team'])
        away_abbr = get_team_abbr(features['away_team'])
        
        if not home_abbr or not away_abbr:
            logger.warning(f"Could not find team abbreviations for {features['home_team']} vs {features['away_team']}")
            return features
        
        # Calculate team features
        home_features = self.calculate_team_features(
            home_abbr, 
            features['season'],
            features['game_date']
        )
        away_features = self.calculate_team_features(
            away_abbr,
            features['season'],
            features['game_date']
        )
        
        # Add home team features
        for key, value in home_features.items():
            features[f'home_{key}'] = value
        
        # Add away team features
        for key, value in away_features.items():
            features[f'away_{key}'] = value
        
        # Calculate differentials
        features['ppg_diff'] = home_features.get('ppg', 0) - away_features.get('ppg', 0)
        features['win_pct_diff'] = home_features.get('win_pct', 0.5) - away_features.get('win_pct', 0.5)
        features['rest_days_diff'] = home_features.get('rest_days', 2) - away_features.get('rest_days', 2)
        
        # Game results
        features['home_score'] = safe_int(game_row.get('home_score', 0))
        features['away_score'] = safe_int(game_row.get('away_score', 0))
        features['home_team_wins'] = 1 if features['home_score'] > features['away_score'] else 0
        
        # Spread (if available)
        features['spread'] = safe_float(game_row.get('spread', 0))
        
        # Injury features
        if self.injury_engine:
            try:
                injury_features = self.injury_engine.get_game_injury_features(
                    home_abbr, away_abbr, features['game_date']
                )
                features.update(injury_features)
            except Exception as e:
                logger.warning(f"Could not get injury features: {e}")
        
        return features

