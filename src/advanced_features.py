"""
Advanced Feature Engineering using nflreadpy

Extracts rich features from:
- Play-by-Play data (EPA, success rate, drive efficiency)
- Next Gen Stats (advanced passing/rushing metrics)
- Team Stats (official NFL statistics)
- Injury Data (player availability, impact scores)
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from datetime import datetime

try:
    import nflreadpy as nfl
except ImportError:
    nfl = None

try:
    from .config import MODELS_DIR
    from .utils import logger, print_section_header
    from .injury_features import InjuryFeatureEngine
except ImportError:
    from config import MODELS_DIR
    from utils import logger, print_section_header
    from injury_features import InjuryFeatureEngine

logger = logging.getLogger(__name__)


class AdvancedFeatureEngine:
    """Extract advanced features from nflreadpy data"""
    
    def __init__(self, seasons: List[int] = None, include_injuries: bool = True):
        """Initialize with seasons to load"""
        if seasons is None:
            # PBP data available: 2015-2024, NGS data available: 2016-2024
            seasons = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
        self.seasons = seasons
        self.include_injuries = include_injuries
        self.pbp_df = None
        self.schedules_df = None
        self.team_stats_df = None
        self.ngs_passing_df = None
        self.ngs_rushing_df = None
        self.ngs_receiving_df = None
        
        # Injury feature engine
        self.injury_engine = None
        if self.include_injuries:
            self.injury_engine = InjuryFeatureEngine(seasons=seasons)
        
    def load_all_data(self):
        """Load all data sources from nflreadpy"""
        print_section_header("Loading Advanced NFL Data")
        
        if nfl is None:
            raise ImportError("nflreadpy not installed. Run: pip install nflreadpy")
        
        print(f"📊 Loading data for seasons: {self.seasons}")
        
        # Play-by-Play
        print("\n1️⃣ Loading Play-by-Play data...")
        pbp = nfl.load_pbp(self.seasons)
        self.pbp_df = pbp.to_pandas()
        print(f"   ✅ Loaded {len(self.pbp_df):,} plays")
        
        # Schedules
        print("\n2️⃣ Loading Schedules...")
        schedules = nfl.load_schedules(self.seasons)
        self.schedules_df = schedules.to_pandas()
        print(f"   ✅ Loaded {len(self.schedules_df)} games")
        
        # Team Stats
        print("\n3️⃣ Loading Team Stats...")
        team_stats = nfl.load_team_stats(self.seasons, summary_level='week')
        self.team_stats_df = team_stats.to_pandas()
        print(f"   ✅ Loaded {len(self.team_stats_df)} team-week records")
        
        # Next Gen Stats - Passing
        print("\n4️⃣ Loading Next Gen Stats (Passing)...")
        try:
            ngs_pass = nfl.load_nextgen_stats(self.seasons, stat_type='passing')
            self.ngs_passing_df = ngs_pass.to_pandas()
            print(f"   ✅ Loaded {len(self.ngs_passing_df)} passing records")
        except Exception as e:
            print(f"   ⚠️ NGS Passing not available: {e}")
            self.ngs_passing_df = None
        
        # Next Gen Stats - Rushing
        print("\n5️⃣ Loading Next Gen Stats (Rushing)...")
        try:
            ngs_rush = nfl.load_nextgen_stats(self.seasons, stat_type='rushing')
            self.ngs_rushing_df = ngs_rush.to_pandas()
            print(f"   ✅ Loaded {len(self.ngs_rushing_df)} rushing records")
        except Exception as e:
            print(f"   ⚠️ NGS Rushing not available: {e}")
            self.ngs_rushing_df = None
        
        # Next Gen Stats - Receiving
        print("\n6️⃣ Loading Next Gen Stats (Receiving)...")
        try:
            ngs_rec = nfl.load_nextgen_stats(self.seasons, stat_type='receiving')
            self.ngs_receiving_df = ngs_rec.to_pandas()
            print(f"   ✅ Loaded {len(self.ngs_receiving_df)} receiving records")
        except Exception as e:
            print(f"   ⚠️ NGS Receiving not available: {e}")
            self.ngs_receiving_df = None
        
        # Injury Data
        if self.include_injuries and self.injury_engine:
            print("\n7️⃣ Loading Injury Data...")
            try:
                self.injury_engine.load_all_data()
                print(f"   ✅ Injury data loaded successfully")
            except Exception as e:
                print(f"   ⚠️ Injury data not available: {e}")
                self.injury_engine = None
        
        print("\n✅ All data loaded successfully!\n")
    
    def spread_to_ml_odds(self, spread: float):
        """
        Convert spread to ML odds using simplified approximation formula
        For favorites (spread < 0): ML = -100 * (1 + 0.12 * ABS(spread))^2 + 100
        For underdogs (spread > 0): ML = 100 * (1 + 0.12 * spread)^2 - 100
        
        Returns: (home_ml_odds, away_ml_odds)
        Spread sign convention: negative = home favored, positive = away favored
        """
        if spread is None or pd.isna(spread):
            return None, None
        
        spread = float(spread)
        
        if spread < 0:
            # Home team is favorite
            favorite_ml = -100 * (1 + 0.12 * abs(spread))**2 + 100
            underdog_ml = 100 * (1 + 0.12 * abs(spread))**2 - 100
            return round(favorite_ml), round(underdog_ml)
        elif spread > 0:
            # Away team is favorite
            favorite_ml = -100 * (1 + 0.12 * abs(spread))**2 + 100
            underdog_ml = 100 * (1 + 0.12 * abs(spread))**2 - 100
            return round(underdog_ml), round(favorite_ml)
        else:
            # Pick'em game
            return 100, -100
    
    def american_odds_to_implied_prob(self, odds):
        """Convert American odds to implied probability"""
        if odds is None or pd.isna(odds):
            return None
        try:
            odds = float(odds)
        except (TypeError, ValueError):
            return None
        
        if odds >= 0:
            return 100.0 / (odds + 100.0)
        else:
            return -odds / (-odds + 100.0)
    
    def calculate_pbp_features(self, team: str, season: int, week: int = None) -> Dict:
        """
        Calculate play-by-play features for a team
        
        Features:
        - EPA (Expected Points Added) per play
        - Success rate (positive EPA plays)
        - Explosive play rate (EPA > threshold)
        - Red zone efficiency
        - Drive success rate
        - Turnover rate
        - 3rd down conversion rate
        - 4th down conversion rate
        - Recent form (last 3 games)
        - Home/away splits
        """
        
        # Filter PBP for this team - CRITICAL: Use < week to avoid data leakage
        if week:
            team_pbp = self.pbp_df[
                (self.pbp_df['season'] == season) &
                (self.pbp_df['week'] < week) &  # < week, NOT <= week (no leakage!)
                ((self.pbp_df['posteam'] == team) | (self.pbp_df['defteam'] == team))
            ].copy()
        else:
            team_pbp = self.pbp_df[
                (self.pbp_df['season'] == season) &
                ((self.pbp_df['posteam'] == team) | (self.pbp_df['defteam'] == team))
            ].copy()
        
        if len(team_pbp) == 0:
            return self._get_default_pbp_features()
        
        # Offensive plays
        off_pbp = team_pbp[team_pbp['posteam'] == team].copy()
        def_pbp = team_pbp[team_pbp['defteam'] == team].copy()
        
        # Home/Away splits
        if 'home_team' in off_pbp.columns:
            # Home games (team is home)
            off_home = off_pbp[off_pbp['home_team'] == team]
            def_home = def_pbp[def_pbp['home_team'] == team]
            
            # Away games (team is away)
            off_away = off_pbp[off_pbp['away_team'] == team]
            def_away = def_pbp[def_pbp['away_team'] == team]
        else:
            off_home = off_pbp
            off_away = off_pbp
            def_home = def_pbp
            def_away = def_pbp
        
        features = {}
        
        # Offensive EPA
        if 'epa' in off_pbp.columns and off_pbp['epa'].notna().any():
            features['off_epa_per_play'] = off_pbp['epa'].mean()
            features['off_success_rate'] = (off_pbp['epa'] > 0).mean()
            features['off_explosive_play_rate'] = (off_pbp['epa'] > 1.0).mean()
        else:
            features['off_epa_per_play'] = 0.0
            features['off_success_rate'] = 0.5
            features['off_explosive_play_rate'] = 0.1
        
        # Defensive EPA (lower is better)
        if 'epa' in def_pbp.columns and def_pbp['epa'].notna().any():
            features['def_epa_per_play'] = def_pbp['epa'].mean()
            features['def_success_rate'] = (def_pbp['epa'] > 0).mean()
        else:
            features['def_epa_per_play'] = 0.0
            features['def_success_rate'] = 0.5
        
        # Red Zone efficiency
        if 'yardline_100' in off_pbp.columns:
            red_zone = off_pbp[off_pbp['yardline_100'] <= 20]
            if len(red_zone) > 0:
                features['off_red_zone_plays'] = len(red_zone)
                if 'touchdown' in red_zone.columns:
                    features['off_red_zone_td_rate'] = red_zone['touchdown'].mean()
                else:
                    features['off_red_zone_td_rate'] = 0.2
            else:
                features['off_red_zone_plays'] = 0
                features['off_red_zone_td_rate'] = 0.2
        else:
            features['off_red_zone_plays'] = 0
            features['off_red_zone_td_rate'] = 0.2
        
        # 3rd down conversions
        if 'down' in off_pbp.columns:
            third_downs = off_pbp[off_pbp['down'] == 3]
            if len(third_downs) > 0 and 'third_down_converted' in third_downs.columns:
                features['off_3rd_down_conv_rate'] = third_downs['third_down_converted'].mean()
            else:
                features['off_3rd_down_conv_rate'] = 0.4
        else:
            features['off_3rd_down_conv_rate'] = 0.4
        
        # Turnovers
        if 'interception' in off_pbp.columns and 'fumble_lost' in off_pbp.columns:
            features['off_turnover_rate'] = (
                off_pbp['interception'].sum() + off_pbp['fumble_lost'].sum()
            ) / len(off_pbp) if len(off_pbp) > 0 else 0.02
            
            features['def_turnover_rate'] = (
                def_pbp['interception'].sum() + def_pbp['fumble_lost'].sum()
            ) / len(def_pbp) if len(def_pbp) > 0 else 0.02
        else:
            features['off_turnover_rate'] = 0.02
            features['def_turnover_rate'] = 0.02
        
        # Yards per play
        if 'yards_gained' in off_pbp.columns:
            features['off_yards_per_play'] = off_pbp['yards_gained'].mean()
            features['def_yards_per_play'] = def_pbp['yards_gained'].mean()
        else:
            features['off_yards_per_play'] = 5.0
            features['def_yards_per_play'] = 5.0
        
        # Pass vs Run ratios
        if 'play_type' in off_pbp.columns:
            pass_plays = off_pbp[off_pbp['play_type'] == 'pass']
            run_plays = off_pbp[off_pbp['play_type'] == 'run']
            total_plays = len(pass_plays) + len(run_plays)
            
            if total_plays > 0:
                features['off_pass_rate'] = len(pass_plays) / total_plays
                features['off_run_rate'] = len(run_plays) / total_plays
            else:
                features['off_pass_rate'] = 0.6
                features['off_run_rate'] = 0.4
        else:
            features['off_pass_rate'] = 0.6
            features['off_run_rate'] = 0.4
        
        # HOME/AWAY SPLITS
        if len(off_home) > 0 and 'epa' in off_home.columns:
            features['off_epa_at_home'] = off_home['epa'].mean()
            features['off_success_at_home'] = (off_home['epa'] > 0).mean()
        else:
            features['off_epa_at_home'] = features['off_epa_per_play']
            features['off_success_at_home'] = features['off_success_rate']
        
        if len(off_away) > 0 and 'epa' in off_away.columns:
            features['off_epa_on_road'] = off_away['epa'].mean()
            features['off_success_on_road'] = (off_away['epa'] > 0).mean()
        else:
            features['off_epa_on_road'] = features['off_epa_per_play']
            features['off_success_on_road'] = features['off_success_rate']
        
        # RECENT FORM (Last 3 games)
        if week and week > 3:
            recent_pbp = self.pbp_df[
                (self.pbp_df['season'] == season) &
                (self.pbp_df['week'] >= week - 3) &
                (self.pbp_df['week'] < week) &
                (self.pbp_df['posteam'] == team)
            ]
            
            if len(recent_pbp) > 0 and 'epa' in recent_pbp.columns:
                recent_epa = recent_pbp['epa'].mean()
                # Trend: positive = improving, negative = declining
                features['recent_form_epa'] = recent_epa
                features['epa_trend'] = recent_epa - features['off_epa_per_play']
            else:
                features['recent_form_epa'] = features['off_epa_per_play']
                features['epa_trend'] = 0.0
        else:
            features['recent_form_epa'] = features['off_epa_per_play']
            features['epa_trend'] = 0.0
        
        return features
    
    def _get_default_pbp_features(self) -> Dict:
        """Default PBP features when no data available"""
        return {
            'off_epa_per_play': 0.0,
            'off_success_rate': 0.5,
            'off_explosive_play_rate': 0.1,
            'def_epa_per_play': 0.0,
            'def_success_rate': 0.5,
            'off_red_zone_plays': 0,
            'off_red_zone_td_rate': 0.2,
            'off_3rd_down_conv_rate': 0.4,
            'off_turnover_rate': 0.02,
            'def_turnover_rate': 0.02,
            'off_yards_per_play': 5.0,
            'def_yards_per_play': 5.0,
            'off_pass_rate': 0.6,
            'off_run_rate': 0.4,
            # Home/away splits
            'off_epa_at_home': 0.0,
            'off_success_at_home': 0.5,
            'off_epa_on_road': 0.0,
            'off_success_on_road': 0.5,
            # Recent form
            'recent_form_epa': 0.0,
            'epa_trend': 0.0,
        }
    
    def calculate_ngs_features(self, team: str, season: int, week: int = None) -> Dict:
        """
        Calculate Next Gen Stats features
        
        Features:
        - Average time to throw
        - Average completed air yards
        - Average separation on targets
        - Average rush yards before contact
        - Efficiency metrics
        """
        features = {}
        
        # NGS Passing
        if self.ngs_passing_df is not None:
            # Check column names - could be 'team_abbr', 'team', or 'team_abb'
            team_col = None
            for col in ['team_abbr', 'team', 'team_abb']:
                if col in self.ngs_passing_df.columns:
                    team_col = col
                    break
            
            if team_col and week:
                # CRITICAL: Only use data from weeks BEFORE the current week
                ngs_pass = self.ngs_passing_df[
                    (self.ngs_passing_df['season'] == season) &
                    (self.ngs_passing_df['week'] < week) &  # < week, not <= week
                    (self.ngs_passing_df[team_col] == team)
                ]
            elif team_col:
                ngs_pass = self.ngs_passing_df[
                    (self.ngs_passing_df['season'] == season) &
                    (self.ngs_passing_df[team_col] == team)
                ]
            else:
                ngs_pass = pd.DataFrame()
            
            if len(ngs_pass) > 0:
                if 'avg_time_to_throw' in ngs_pass.columns:
                    features['avg_time_to_throw'] = ngs_pass['avg_time_to_throw'].mean()
                if 'avg_completed_air_yards' in ngs_pass.columns:
                    features['avg_completed_air_yards'] = ngs_pass['avg_completed_air_yards'].mean()
                if 'avg_air_yards_differential' in ngs_pass.columns:
                    features['avg_air_yards_diff'] = ngs_pass['avg_air_yards_differential'].mean()
                if 'aggressiveness' in ngs_pass.columns:
                    features['pass_aggressiveness'] = ngs_pass['aggressiveness'].mean()
        
        # NGS Rushing
        if self.ngs_rushing_df is not None:
            # Check column names
            team_col = None
            for col in ['team_abbr', 'team', 'team_abb']:
                if col in self.ngs_rushing_df.columns:
                    team_col = col
                    break
            
            if team_col and week:
                # CRITICAL: Only use data from weeks BEFORE the current week
                ngs_rush = self.ngs_rushing_df[
                    (self.ngs_rushing_df['season'] == season) &
                    (self.ngs_rushing_df['week'] < week) &  # < week, not <= week
                    (self.ngs_rushing_df[team_col] == team)
                ]
            elif team_col:
                ngs_rush = self.ngs_rushing_df[
                    (self.ngs_rushing_df['season'] == season) &
                    (self.ngs_rushing_df[team_col] == team)
                ]
            else:
                ngs_rush = pd.DataFrame()
            
            if len(ngs_rush) > 0:
                if 'avg_rush_yards' in ngs_rush.columns:
                    features['avg_rush_yards_ngs'] = ngs_rush['avg_rush_yards'].mean()
                if 'avg_time_to_los' in ngs_rush.columns:
                    features['avg_time_to_los'] = ngs_rush['avg_time_to_los'].mean()
                if 'rush_attempts' in ngs_rush.columns:
                    features['rush_attempts'] = ngs_rush['rush_attempts'].sum()
        
        # Fill defaults
        features.setdefault('avg_time_to_throw', 2.5)
        features.setdefault('avg_completed_air_yards', 6.0)
        features.setdefault('avg_air_yards_diff', 0.0)
        features.setdefault('pass_aggressiveness', 0.1)
        features.setdefault('avg_rush_yards_ngs', 4.0)
        features.setdefault('avg_time_to_los', 2.5)
        features.setdefault('rush_attempts', 25)
        
        return features
    
    def calculate_team_stats_features(self, team: str, season: int, week: int = None) -> Dict:
        """
        Calculate official team stats features
        
        Features from official NFL stats:
        - Passing yards, TDs, INTs
        - Rushing yards, TDs
        - Total yards
        - Sacks taken/given
        - Penalties
        """
        features = {}
        
        if self.team_stats_df is not None:
            # Check column names
            team_col = None
            for col in ['team_abbr', 'team', 'team_abb']:
                if col in self.team_stats_df.columns:
                    team_col = col
                    break
            
            if team_col and week:
                # CRITICAL: Only use data from weeks BEFORE the current week
                team_data = self.team_stats_df[
                    (self.team_stats_df['season'] == season) &
                    (self.team_stats_df['week'] < week) &  # < week, not <= week
                    (self.team_stats_df[team_col] == team)
                ]
            elif team_col:
                team_data = self.team_stats_df[
                    (self.team_stats_df['season'] == season) &
                    (self.team_stats_df[team_col] == team)
                ]
            else:
                team_data = pd.DataFrame()
            
            if len(team_data) > 0:
                # Passing
                if 'passing_yards' in team_data.columns:
                    features['passing_yards_per_game'] = team_data['passing_yards'].mean()
                if 'passing_tds' in team_data.columns:
                    features['passing_tds_per_game'] = team_data['passing_tds'].mean()
                if 'interceptions' in team_data.columns:
                    features['interceptions_per_game'] = team_data['interceptions'].mean()
                
                # Rushing
                if 'rushing_yards' in team_data.columns:
                    features['rushing_yards_per_game'] = team_data['rushing_yards'].mean()
                if 'rushing_tds' in team_data.columns:
                    features['rushing_tds_per_game'] = team_data['rushing_tds'].mean()
                
                # Sacks
                if 'sacks' in team_data.columns:
                    features['sacks_taken_per_game'] = team_data['sacks'].mean()
                
                # Penalties
                if 'penalties' in team_data.columns:
                    features['penalties_per_game'] = team_data['penalties'].mean()
        
        # Fill defaults
        features.setdefault('passing_yards_per_game', 220.0)
        features.setdefault('passing_tds_per_game', 1.5)
        features.setdefault('interceptions_per_game', 1.0)
        features.setdefault('rushing_yards_per_game', 110.0)
        features.setdefault('rushing_tds_per_game', 1.0)
        features.setdefault('sacks_taken_per_game', 2.5)
        features.setdefault('penalties_per_game', 6.0)
        
        return features
    
    def get_all_features_for_game(
        self, 
        home_team: str, 
        away_team: str, 
        season: int, 
        week: int
    ) -> Dict:
        """
        Get all advanced features for a game
        
        Returns dict with home_* and away_* features
        """
        print(f"\n🔍 Extracting features: {away_team} @ {home_team} (Week {week}, {season})")
        
        # Home team features
        home_pbp = self.calculate_pbp_features(home_team, season, week)
        home_ngs = self.calculate_ngs_features(home_team, season, week)
        home_stats = self.calculate_team_stats_features(home_team, season, week)
        
        # Away team features
        away_pbp = self.calculate_pbp_features(away_team, season, week)
        away_ngs = self.calculate_ngs_features(away_team, season, week)
        away_stats = self.calculate_team_stats_features(away_team, season, week)
        
        # Combine with prefixes
        all_features = {}
        
        for key, val in home_pbp.items():
            all_features[f'home_{key}'] = val
        for key, val in home_ngs.items():
            all_features[f'home_{key}'] = val
        for key, val in home_stats.items():
            all_features[f'home_{key}'] = val
            
        for key, val in away_pbp.items():
            all_features[f'away_{key}'] = val
        for key, val in away_ngs.items():
            all_features[f'away_{key}'] = val
        for key, val in away_stats.items():
            all_features[f'away_{key}'] = val
        
        # Add injury features if available
        if self.include_injuries and self.injury_engine:
            try:
                injury_features = self.injury_engine.get_injury_features_for_game(
                    home_team, away_team, season, week
                )
                all_features.update(injury_features)
                
                # Calculate injury-adjusted EPA
                all_features['home_injury_adj_off_epa'] = (
                    all_features['home_off_epa_per_play'] * 
                    (1 - all_features['home_off_injury_impact'])
                )
                all_features['away_injury_adj_off_epa'] = (
                    all_features['away_off_epa_per_play'] * 
                    (1 - all_features['away_off_injury_impact'])
                )
                
                all_features['home_injury_adj_def_epa'] = (
                    all_features['home_def_epa_per_play'] * 
                    (1 + all_features['home_def_injury_impact'])
                )
                all_features['away_injury_adj_def_epa'] = (
                    all_features['away_def_epa_per_play'] * 
                    (1 + all_features['away_def_injury_impact'])
                )
                
            except Exception as e:
                logger.warning(f"Could not extract injury features: {e}")
        
        print(f"   ✅ Extracted {len(all_features)} advanced features")
        
        return all_features
    
    def calculate_rest_days(self, schedules_df: pd.DataFrame) -> Dict:
        """
        Calculate rest days for each team before each game
        Returns dict: {(team, season, week): days_rest}
        """
        print("📊 Calculating rest days...")
        
        # Sort by team, season, date
        schedules = schedules_df.copy()
        schedules['gameday'] = pd.to_datetime(schedules['gameday'])
        schedules = schedules.sort_values(['season', 'gameday'])
        
        rest_days = {}
        last_game_date = {}
        
        for _, game in schedules.iterrows():
            game_date = game['gameday']
            season = game['season']
            week = game['week']
            home_team = game['home_team']
            away_team = game['away_team']
            
            for team in [home_team, away_team]:
                team_key = (team, season)
                
                if team_key in last_game_date:
                    days = (game_date - last_game_date[team_key]).days
                    rest_days[(team, season, week)] = days
                else:
                    # First game of season = standard 7 days
                    rest_days[(team, season, week)] = 7
                
                # Update last game date for this team
                last_game_date[team_key] = game_date
        
        print(f"   ✅ Calculated rest days for {len(rest_days)} team-week combinations")
        return rest_days
    
    def calculate_team_records(self, schedules_df: pd.DataFrame) -> Dict:
        """
        Calculate W/L records for each team up to each game (no leakage)
        Returns dict: {(team, season, week): {'wins': x, 'losses': y, 'win_pct': z}}
        """
        print("📊 Calculating team W/L records...")
        
        # Sort by season and week
        schedules = schedules_df.sort_values(['season', 'week']).copy()
        
        team_records = {}
        
        # Track cumulative records for each team by season
        for season in schedules['season'].unique():
            season_games = schedules[schedules['season'] == season].copy()
            
            # Initialize records for all teams in this season
            teams = pd.concat([season_games['home_team'], season_games['away_team']]).unique()
            season_records = {team: {'wins': 0, 'losses': 0} for team in teams}
            
            for _, game in season_games.iterrows():
                week = game['week']
                home_team = game['home_team']
                away_team = game['away_team']
                
                # Store CURRENT record BEFORE this game (no leakage)
                for team in [home_team, away_team]:
                    rec = season_records[team]
                    total_games = rec['wins'] + rec['losses']
                    win_pct = rec['wins'] / total_games if total_games > 0 else 0.5
                    
                    team_records[(team, season, week)] = {
                        'wins': rec['wins'],
                        'losses': rec['losses'],
                        'win_pct': win_pct
                    }
                
                # Update records AFTER game completes (for next game)
                if pd.notna(game.get('home_score')) and pd.notna(game.get('away_score')):
                    home_score = game['home_score']
                    away_score = game['away_score']
                    
                    if home_score > away_score:
                        season_records[home_team]['wins'] += 1
                        season_records[away_team]['losses'] += 1
                    else:
                        season_records[away_team]['wins'] += 1
                        season_records[home_team]['losses'] += 1
        
        print(f"   ✅ Calculated records for {len(team_records)} team-week combinations")
        return team_records
    
    def create_enhanced_dataset(self, output_path: str = None):
        """
        Create enhanced dataset with all advanced features
        """
        print_section_header("Creating Enhanced Dataset with Advanced Features")
        
        if output_path is None:
            from pathlib import Path
            output_path = Path(__file__).parent.parent / 'data' / 'enhanced_features.csv'
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load all data
        self.load_all_data()
        
        # Calculate team records and rest days (before processing games to avoid leakage)
        team_records = self.calculate_team_records(self.schedules_df)
        rest_days_dict = self.calculate_rest_days(self.schedules_df)
        
        # Process all games
        enhanced_data = []
        
        for idx, game in self.schedules_df.iterrows():
            if pd.isna(game['home_score']):
                continue  # Skip incomplete games
            
            try:
                features = self.get_all_features_for_game(
                    home_team=game['home_team'],
                    away_team=game['away_team'],
                    season=game['season'],
                    week=game['week']
                )
                
                # Add W/L records BEFORE this game (no leakage)
                home_record = team_records.get((game['home_team'], game['season'], game['week']), 
                                               {'wins': 0, 'losses': 0, 'win_pct': 0.5})
                away_record = team_records.get((game['away_team'], game['season'], game['week']), 
                                               {'wins': 0, 'losses': 0, 'win_pct': 0.5})
                
                features['home_wins'] = home_record['wins']
                features['home_losses'] = home_record['losses']
                features['home_win_pct'] = home_record['win_pct']
                features['away_wins'] = away_record['wins']
                features['away_losses'] = away_record['losses']
                features['away_win_pct'] = away_record['win_pct']
                features['record_diff'] = home_record['win_pct'] - away_record['win_pct']
                
                # Add rest days
                home_rest = rest_days_dict.get((game['home_team'], game['season'], game['week']), 7)
                away_rest = rest_days_dict.get((game['away_team'], game['season'], game['week']), 7)
                features['home_rest_days'] = home_rest
                features['away_rest_days'] = away_rest
                features['rest_advantage'] = home_rest - away_rest
                
                # Add game info
                features['game_id'] = game.get('game_id', f"{game['season']}_{game['week']}_{game['away_team']}_{game['home_team']}")
                features['season'] = game['season']
                features['week'] = game['week']
                features['home_team'] = game['home_team']
                features['away_team'] = game['away_team']
                features['home_score'] = game['home_score']
                features['away_score'] = game['away_score']
                features['home_team_wins'] = 1 if game['home_score'] > game['away_score'] else 0
                
                # Spread if available
                if 'spread_line' in game and not pd.isna(game['spread_line']):
                    features['spread'] = game['spread_line']
                    score_diff = game['home_score'] - game['away_score']
                    features['ats_home_covers'] = 1 if (score_diff + game['spread_line']) > 0 else 0
                    
                    # Calculate market ML odds from spread using approximation formula
                    spread_value = game['spread_line']
                    market_home_ml, market_away_ml = self.spread_to_ml_odds(spread_value)
                    
                    # Convert to implied probabilities
                    if market_home_ml is not None and market_away_ml is not None:
                        home_implied = self.american_odds_to_implied_prob(market_home_ml)
                        away_implied = self.american_odds_to_implied_prob(market_away_ml)
                        
                        if home_implied is not None and away_implied is not None:
                            # Normalize to sum to 1.0
                            total = home_implied + away_implied
                            if total > 0:
                                features['market_home_ml_prob'] = home_implied / total
                                features['market_away_ml_prob'] = away_implied / total
                                features['market_prob_diff'] = features['market_home_ml_prob'] - features['market_away_ml_prob']
                            else:
                                features['market_home_ml_prob'] = 0.5
                                features['market_away_ml_prob'] = 0.5
                                features['market_prob_diff'] = 0.0
                        else:
                            features['market_home_ml_prob'] = 0.5
                            features['market_away_ml_prob'] = 0.5
                            features['market_prob_diff'] = 0.0
                    else:
                        features['market_home_ml_prob'] = 0.5
                        features['market_away_ml_prob'] = 0.5
                        features['market_prob_diff'] = 0.0
                else:
                    features['spread'] = 0
                    features['ats_home_covers'] = features['home_team_wins']
                    features['market_home_ml_prob'] = 0.5
                    features['market_away_ml_prob'] = 0.5
                    features['market_prob_diff'] = 0.0
                
                enhanced_data.append(features)
                
                if len(enhanced_data) % 50 == 0:
                    print(f"   Processed {len(enhanced_data)} games...")
                
            except Exception as e:
                logger.warning(f"Error processing game {idx}: {e}")
                continue
        
        # Create DataFrame
        enhanced_df = pd.DataFrame(enhanced_data)
        
        # Save
        enhanced_df.to_csv(output_path, index=False)
        
        print_section_header("Enhanced Dataset Created! 🎉")
        print(f"✅ Total games: {len(enhanced_df)}")
        print(f"✅ Total features: {len(enhanced_df.columns)}")
        print(f"✅ Saved to: {output_path}")
        
        # Print feature breakdown
        pbp_features = len([c for c in enhanced_df.columns if any(x in c for x in ['epa', 'success', 'explosive', 'red_zone', '3rd_down', 'turnover', 'yards_per_play'])])
        ngs_features = len([c for c in enhanced_df.columns if any(x in c for x in ['time_to_throw', 'air_yards', 'aggressiveness', 'time_to_los'])])
        stats_features = len([c for c in enhanced_df.columns if any(x in c for x in ['passing_yards', 'rushing_yards', 'sacks', 'penalties'])])
        injury_features = len([c for c in enhanced_df.columns if any(x in c for x in ['injury', 'qb_out', 'wr_out', 'cb_out', 'lt_out', 'edge_out', 'ol_out'])])
        
        print(f"\n📊 Feature Breakdown:")
        print(f"   Play-by-Play: {pbp_features} features")
        print(f"   Next Gen Stats: {ngs_features} features")
        print(f"   Team Stats: {stats_features} features")
        if self.include_injuries:
            print(f"   Injury Features: {injury_features} features")
        
        return enhanced_df


def main():
    """Run advanced feature engineering"""
    engine = AdvancedFeatureEngine(seasons=[2024, 2025])
    engine.create_enhanced_dataset()


if __name__ == "__main__":
    main()

