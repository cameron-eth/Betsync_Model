"""
Current Season Feature Fetcher
Uses nflreadpy to fetch real current season stats up to current week -1
Includes injury data for current week predictions
"""

import nflreadpy as nfl
import polars as pl
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple
import logging

try:
    from .injury_features import InjuryFeatureEngine
except ImportError:
    from injury_features import InjuryFeatureEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CurrentSeasonFeatures:
    """Fetch and calculate current season features for predictions"""
    
    def __init__(self, current_season: int = 2025, current_week: int = 8, include_injuries: bool = True):
        self.current_season = current_season
        self.current_week = current_week
        self.include_injuries = include_injuries
        self.team_features = None
        self.injury_engine = None
        self.schedules_df = None
        
        # Initialize injury engine if enabled
        if self.include_injuries:
            self.injury_engine = InjuryFeatureEngine(seasons=[current_season])
        
    def fetch_current_season_data(self):
        """Fetch play-by-play data up to current week -1"""
        logger.info(f"Fetching {self.current_season} season data up to week {self.current_week}...")
        
        try:
            # Load current season play-by-play data
            pbp = nfl.load_pbp([self.current_season])
            
            # Convert to pandas for easier manipulation
            pbp_df = pbp.to_pandas()
            
            # Filter to only weeks before current week (avoid leakage)
            # If current_week is 12, use weeks 1-11
            pbp_df = pbp_df[pbp_df['week'] < self.current_week]
            
            logger.info(f"Loaded {len(pbp_df)} plays from weeks 1-{self.current_week - 1}")
            
            # Also load schedules for rest days calculation
            schedules = nfl.load_schedules([self.current_season])
            self.schedules_df = schedules.to_pandas()
            logger.info(f"Loaded {len(self.schedules_df)} scheduled games")
            
            return pbp_df
            
        except Exception as e:
            logger.error(f"Error fetching play-by-play data: {e}")
            return None
    
    def calculate_team_features(self, pbp_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate team-level features from play-by-play data"""
        logger.info("Calculating team features from play-by-play data...")
        
        team_features = []
        
        # Get unique teams
        teams = pd.concat([pbp_df['posteam'], pbp_df['defteam']]).dropna().unique()
        
        # Count games per team for per-game calculations
        games_per_team = pbp_df.groupby('posteam')['game_id'].nunique()
        
        # Calculate win-loss records
        # Get final scores for each game
        game_results = pbp_df.groupby('game_id').agg({
            'home_team': 'first',
            'away_team': 'first',
            'total_home_score': 'max',
            'total_away_score': 'max'
        }).reset_index()
        
        # Calculate wins for each team
        team_records = {}
        for team in teams:
            home_wins = len(game_results[(game_results['home_team'] == team) & 
                                         (game_results['total_home_score'] > game_results['total_away_score'])])
            away_wins = len(game_results[(game_results['away_team'] == team) & 
                                         (game_results['total_away_score'] > game_results['total_home_score'])])
            
            home_losses = len(game_results[(game_results['home_team'] == team) & 
                                           (game_results['total_home_score'] < game_results['total_away_score'])])
            away_losses = len(game_results[(game_results['away_team'] == team) & 
                                           (game_results['total_away_score'] < game_results['total_home_score'])])
            
            total_wins = home_wins + away_wins
            total_losses = home_losses + away_losses
            win_pct = total_wins / (total_wins + total_losses) if (total_wins + total_losses) > 0 else 0.5
            
            team_records[team] = {
                'wins': total_wins,
                'losses': total_losses,
                'win_pct': win_pct
            }
        
        for team in teams:
            # Offensive stats (when team has possession)
            offense = pbp_df[pbp_df['posteam'] == team].copy()
            
            # Defensive stats (when team is on defense)
            defense = pbp_df[pbp_df['defteam'] == team].copy()
            
            num_games = games_per_team.get(team, 1)
            
            # Helper function for safe mean calculation
            def safe_mean(series, default=0.0):
                return series.mean() if len(series) > 0 and series.notna().any() else default
            
            # Calculate home/away splits
            home_games = offense[offense['home_team'] == team] if 'home_team' in offense.columns else pd.DataFrame()
            away_games = offense[offense['away_team'] == team] if 'away_team' in offense.columns else pd.DataFrame()
            
            # Recent form (last 3 weeks) - use current_week - 1
            if 'week' in offense.columns and self.current_week > 3:
                recent_games = offense[offense['week'] >= self.current_week - 4]
            else:
                recent_games = offense
            
            # Get team record
            record = team_records.get(team, {'wins': 0, 'losses': 0, 'win_pct': 0.5})
            
            features = {
                'team': team,
                'season': self.current_season,
                
                # Team Record
                'wins': record['wins'],
                'losses': record['losses'],
                'win_pct': record['win_pct'],
                
                # Offensive EPA metrics
                'off_epa_per_play': safe_mean(offense['epa']),
                'off_success_rate': safe_mean(offense['success']),
                'off_explosive_play_rate': ((offense['yards_gained'] >= 20) & ((offense['pass'] == 1) | (offense['rush'] == 1))).mean() if len(offense) > 0 else 0.0,
                
                # Defensive EPA metrics
                'def_epa_per_play': safe_mean(defense['epa']),
                'def_success_rate': safe_mean(defense['success']),
                
                # Red zone stats
                'off_red_zone_plays': len(offense[offense['yardline_100'] <= 20]),
                'off_red_zone_td_rate': (offense[offense['yardline_100'] <= 20]['touchdown'] == 1).mean() if len(offense[offense['yardline_100'] <= 20]) > 0 else 0.0,
                
                # 3rd down conversion
                'off_3rd_down_conv_rate': (offense[offense['down'] == 3]['third_down_converted'] == 1).mean() if len(offense[offense['down'] == 3]) > 0 else 0.0,
                
                # Turnover rates
                'off_turnover_rate': ((offense['interception'] == 1) | (offense['fumble_lost'] == 1)).mean() if len(offense) > 0 else 0.0,
                'def_turnover_rate': ((defense['interception'] == 1) | (defense['fumble_lost'] == 1)).mean() if len(defense) > 0 else 0.0,
                
                # Yards per play
                'off_yards_per_play': safe_mean(offense['yards_gained'], 5.0),
                'def_yards_per_play': safe_mean(defense['yards_gained'], 5.0),
                
                # Pass/Run rates
                'off_pass_rate': (offense['pass'] == 1).mean() if len(offense) > 0 else 0.5,
                'off_run_rate': (offense['rush'] == 1).mean() if len(offense) > 0 else 0.5,
                
                # NGS metrics (may not be available in all data, use safe defaults)
                'avg_time_to_throw': safe_mean(offense[offense['pass'] == 1]['time_to_throw'] if 'time_to_throw' in offense.columns else pd.Series([]), 2.7),
                'avg_completed_air_yards': safe_mean(offense[(offense['pass'] == 1) & (offense['complete_pass'] == 1)]['air_yards'] if 'air_yards' in offense.columns else pd.Series([]), 6.5),
                'avg_air_yards_diff': safe_mean(offense[(offense['pass'] == 1) & (offense['complete_pass'] == 1)]['yards_after_catch'] if 'yards_after_catch' in offense.columns else pd.Series([]), 4.5),
                'pass_aggressiveness': (safe_mean(offense[offense['pass'] == 1]['air_yards'] if 'air_yards' in offense.columns else pd.Series([]), 8.0) / 10.0) if len(offense[offense['pass'] == 1]) > 0 else 0.8,
                'avg_rush_yards_ngs': safe_mean(offense[offense['rush'] == 1]['yards_gained'], 4.0),
                'avg_time_to_los': 1.5,  # Not typically in PBP data, use default
                'rush_attempts': len(offense[offense['rush'] == 1]),
                
                # Per-game stats
                'passing_yards_per_game': offense[offense['pass'] == 1]['yards_gained'].sum() / num_games,
                'passing_tds_per_game': (offense[(offense['pass'] == 1) & (offense['touchdown'] == 1)]).shape[0] / num_games,
                'rushing_yards_per_game': offense[offense['rush'] == 1]['yards_gained'].sum() / num_games,
                'rushing_tds_per_game': (offense[(offense['rush'] == 1) & (offense['touchdown'] == 1)]).shape[0] / num_games,
                'penalties_per_game': (offense['penalty'] == 1).sum() / num_games if 'penalty' in offense.columns else 1.5,
                'interceptions_per_game': (offense['interception'] == 1).sum() / num_games,
                'sacks_taken_per_game': (offense['sack'] == 1).sum() / num_games,
                
                # HOME/AWAY SPLITS
                'off_epa_at_home': safe_mean(home_games['epa']) if len(home_games) > 0 else safe_mean(offense['epa']),
                'off_success_at_home': safe_mean(home_games['epa'] > 0, 0.5) if len(home_games) > 0 else 0.5,
                'off_epa_on_road': safe_mean(away_games['epa']) if len(away_games) > 0 else safe_mean(offense['epa']),
                'off_success_on_road': safe_mean(away_games['epa'] > 0, 0.5) if len(away_games) > 0 else 0.5,
                
                # RECENT FORM (Last 3 games)
                'recent_form_epa': safe_mean(recent_games['epa']) if len(recent_games) > 0 else safe_mean(offense['epa']),
                'epa_trend': safe_mean(recent_games['epa']) - safe_mean(offense['epa']) if len(recent_games) > 0 else 0.0,
            }
            
            team_features.append(features)
        
        features_df = pd.DataFrame(team_features)
        logger.info(f"Calculated features for {len(features_df)} teams")
        
        return features_df
    
    def calculate_rest_days_for_team(self, team: str, week: int) -> int:
        """Calculate rest days for a team before a specific week"""
        if self.schedules_df is None:
            return 7  # Default
        
        # Get team's games
        team_games = self.schedules_df[
            ((self.schedules_df['home_team'] == team) | (self.schedules_df['away_team'] == team)) &
            (self.schedules_df['week'] < week)
        ].copy()
        
        if len(team_games) == 0:
            return 7
        
        # Get last game
        team_games['gameday'] = pd.to_datetime(team_games['gameday'])
        last_game = team_games.sort_values('gameday').iloc[-1]
        
        # Get current week's game
        current_game = self.schedules_df[
            ((self.schedules_df['home_team'] == team) | (self.schedules_df['away_team'] == team)) &
            (self.schedules_df['week'] == week)
        ]
        
        if len(current_game) > 0:
            current_game['gameday'] = pd.to_datetime(current_game['gameday'])
            current_date = current_game.iloc[0]['gameday']
            last_date = last_game['gameday']
            days_rest = (current_date - last_date).days
            return max(days_rest, 3)  # Minimum 3 days (Thursday games)
        
        return 7
    
    def spread_to_ml_odds(self, spread: float, is_home_favorite: bool = True):
        """
        Convert spread to ML odds using simplified approximation formula
        For favorites (spread < 0): ML = -100 * (1 + 0.12 * ABS(spread))^2 + 100
        For underdogs (spread > 0): ML = 100 * (1 + 0.12 * spread)^2 - 100
        """
        if spread is None:
            return None, None
        
        # Determine which team is favored based on spread sign
        # Negative spread = home team favored, positive = away team favored
        if spread < 0:
            # Home team is favorite
            favorite_ml = -100 * (1 + 0.12 * abs(spread))**2 + 100
            underdog_ml = 100 * (1 + 0.12 * abs(spread))**2 - 100
            home_ml = favorite_ml if is_home_favorite else underdog_ml
            away_ml = underdog_ml if is_home_favorite else favorite_ml
        elif spread > 0:
            # Away team is favorite
            favorite_ml = -100 * (1 + 0.12 * abs(spread))**2 + 100
            underdog_ml = 100 * (1 + 0.12 * abs(spread))**2 - 100
            home_ml = underdog_ml if is_home_favorite else favorite_ml
            away_ml = favorite_ml if is_home_favorite else underdog_ml
        else:
            # Pick'em game
            home_ml = 100
            away_ml = -100
        
        return round(home_ml), round(away_ml)
    
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
    
    def create_matchup_features(self, home_team: str, away_team: str, spread: float, week: int = None, market_home_ml_prob: float = None, market_away_ml_prob: float = None) -> Dict:
        """Create feature vector for a specific matchup with ALL required model features"""
        if self.team_features is None:
            raise ValueError("Team features not loaded. Call load_and_calculate_features() first.")
        
        # Get features for both teams
        home_features = self.team_features[self.team_features['team'] == home_team]
        away_features = self.team_features[self.team_features['team'] == away_team]
        
        if len(home_features) == 0 or len(away_features) == 0:
            logger.warning(f"Missing features for {home_team} or {away_team}, using defaults")
            default_features = self.create_default_features(
                spread, 
                week or self.current_week, 
                market_home_ml_prob, 
                market_away_ml_prob
            )
            # Add injury features even when using defaults
            if self.include_injuries and self.injury_engine:
                try:
                    injury_features = self.injury_engine.get_injury_features_for_game(
                        home_team=home_team,
                        away_team=away_team,
                        season=self.current_season,
                        week=week or self.current_week
                    )
                    default_features.update(injury_features)
                except Exception as e:
                    logger.warning(f"Could not add injury features, using defaults: {e}")
                    default_features.update(self._get_default_injury_features())
            else:
                default_features.update(self._get_default_injury_features())
            return default_features
        
        home_features = home_features.iloc[0]
        away_features = away_features.iloc[0]
        
        # Calculate rest days
        home_rest = self.calculate_rest_days_for_team(home_team, week or self.current_week)
        away_rest = self.calculate_rest_days_for_team(away_team, week or self.current_week)
        
        # Create matchup features matching EXACT model expectations
        matchup_features = {
            'spread': spread,
            
            # TEAM RECORDS (4 features - new!)
            'home_wins': home_features['wins'],
            'home_losses': home_features['losses'],
            'home_win_pct': home_features['win_pct'],
            'record_diff': home_features['win_pct'] - away_features['win_pct'],
            
            # HOME TEAM FEATURES (29 features)
            'home_off_epa_per_play': home_features['off_epa_per_play'],
            'home_off_success_rate': home_features['off_success_rate'],
            'home_off_explosive_play_rate': home_features['off_explosive_play_rate'],
            'home_def_epa_per_play': home_features['def_epa_per_play'],
            'home_def_success_rate': home_features['def_success_rate'],
            'home_off_red_zone_plays': home_features['off_red_zone_plays'],
            'home_off_red_zone_td_rate': home_features['off_red_zone_td_rate'],
            'home_off_3rd_down_conv_rate': home_features['off_3rd_down_conv_rate'],
            'home_off_turnover_rate': home_features['off_turnover_rate'],
            'home_def_turnover_rate': home_features['def_turnover_rate'],
            'home_off_yards_per_play': home_features['off_yards_per_play'],
            'home_def_yards_per_play': home_features['def_yards_per_play'],
            'home_off_pass_rate': home_features['off_pass_rate'],
            'home_off_run_rate': home_features['off_run_rate'],
            'home_avg_time_to_throw': home_features['avg_time_to_throw'],
            'home_avg_completed_air_yards': home_features['avg_completed_air_yards'],
            'home_avg_air_yards_diff': home_features['avg_air_yards_diff'],
            'home_pass_aggressiveness': home_features['pass_aggressiveness'],
            'home_avg_rush_yards_ngs': home_features['avg_rush_yards_ngs'],
            'home_avg_time_to_los': home_features['avg_time_to_los'],
            'home_rush_attempts': home_features['rush_attempts'],
            'home_passing_yards_per_game': home_features['passing_yards_per_game'],
            'home_passing_tds_per_game': home_features['passing_tds_per_game'],
            'home_rushing_yards_per_game': home_features['rushing_yards_per_game'],
            'home_rushing_tds_per_game': home_features['rushing_tds_per_game'],
            'home_penalties_per_game': home_features['penalties_per_game'],
            'home_interceptions_per_game': home_features['interceptions_per_game'],
            'home_sacks_taken_per_game': home_features['sacks_taken_per_game'],
            'home_off_epa_at_home': home_features.get('off_epa_at_home', home_features['off_epa_per_play']),
            'home_off_success_at_home': home_features.get('off_success_at_home', home_features['off_success_rate']),
            'home_off_epa_on_road': home_features.get('off_epa_on_road', home_features['off_epa_per_play']),
            'home_off_success_on_road': home_features.get('off_success_on_road', home_features['off_success_rate']),
            'home_recent_form_epa': home_features.get('recent_form_epa', home_features['off_epa_per_play']),
            'home_epa_trend': home_features.get('epa_trend', 0.0),
            'home_rest_days': home_rest,
            
            'away_wins': away_features['wins'],
            'away_losses': away_features['losses'],
            'away_win_pct': away_features['win_pct'],
            
            'away_off_epa_per_play': away_features['off_epa_per_play'],
            'away_off_success_rate': away_features['off_success_rate'],
            'away_off_explosive_play_rate': away_features['off_explosive_play_rate'],
            'away_def_epa_per_play': away_features['def_epa_per_play'],
            'away_def_success_rate': away_features['def_success_rate'],
            'away_off_red_zone_plays': away_features['off_red_zone_plays'],
            'away_off_red_zone_td_rate': away_features['off_red_zone_td_rate'],
            'away_off_3rd_down_conv_rate': away_features['off_3rd_down_conv_rate'],
            'away_off_turnover_rate': away_features['off_turnover_rate'],
            'away_def_turnover_rate': away_features['def_turnover_rate'],
            'away_off_yards_per_play': away_features['off_yards_per_play'],
            'away_def_yards_per_play': away_features['def_yards_per_play'],
            'away_off_pass_rate': away_features['off_pass_rate'],
            'away_off_run_rate': away_features['off_run_rate'],
            'away_avg_time_to_throw': away_features['avg_time_to_throw'],
            'away_avg_completed_air_yards': away_features['avg_completed_air_yards'],
            'away_avg_air_yards_diff': away_features['avg_air_yards_diff'],
            'away_pass_aggressiveness': away_features['pass_aggressiveness'],
            'away_avg_rush_yards_ngs': away_features['avg_rush_yards_ngs'],
            'away_avg_time_to_los': away_features['avg_time_to_los'],
            'away_rush_attempts': away_features['rush_attempts'],
            'away_passing_yards_per_game': away_features['passing_yards_per_game'],
            'away_passing_tds_per_game': away_features['passing_tds_per_game'],
            'away_rushing_yards_per_game': away_features['rushing_yards_per_game'],
            'away_rushing_tds_per_game': away_features['rushing_tds_per_game'],
            'away_penalties_per_game': away_features['penalties_per_game'],
            'away_interceptions_per_game': away_features['interceptions_per_game'],
            'away_sacks_taken_per_game': away_features['sacks_taken_per_game'],
            'away_off_epa_at_home': away_features.get('off_epa_at_home', away_features['off_epa_per_play']),
            'away_off_success_at_home': away_features.get('off_success_at_home', away_features['off_success_rate']),
            'away_off_epa_on_road': away_features.get('off_epa_on_road', away_features['off_epa_per_play']),
            'away_off_success_on_road': away_features.get('off_success_on_road', away_features['off_success_rate']),
            'away_recent_form_epa': away_features.get('recent_form_epa', away_features['off_epa_per_play']),
            'away_epa_trend': away_features.get('epa_trend', 0.0),
            'away_rest_days': away_rest,
            
            # Rest advantage
            'rest_advantage': home_rest - away_rest,
            
            # Week (1 feature)
            'week': week or self.current_week,
            
            # Market odds features (3 features)
            'market_home_ml_prob': market_home_ml_prob if market_home_ml_prob is not None else (
                self.american_to_implied_prob(self.spread_to_ml_odds(spread)[0]) if spread is not None else 0.5
            ),
            'market_away_ml_prob': market_away_ml_prob if market_away_ml_prob is not None else (
                self.american_to_implied_prob(self.spread_to_ml_odds(spread)[1]) if spread is not None else 0.5
            ),
            'market_prob_diff': (
                (market_home_ml_prob - market_away_ml_prob) if market_home_ml_prob is not None and market_away_ml_prob is not None else (
                    (self.american_to_implied_prob(self.spread_to_ml_odds(spread)[0]) - 
                     self.american_to_implied_prob(self.spread_to_ml_odds(spread)[1])) if spread is not None else 0.0
                )
            ),
        }
        
        # Normalize market probabilities if provided
        if market_home_ml_prob is not None and market_away_ml_prob is not None:
            total = market_home_ml_prob + market_away_ml_prob
            if total > 0:
                matchup_features['market_home_ml_prob'] = market_home_ml_prob / total
                matchup_features['market_away_ml_prob'] = market_away_ml_prob / total
                matchup_features['market_prob_diff'] = matchup_features['market_home_ml_prob'] - matchup_features['market_away_ml_prob']
        
        # Add injury features - ALWAYS add them (even if 0.0) to match training
        if self.include_injuries and self.injury_engine:
            try:
                injury_features = self.injury_engine.get_injury_features_for_game(
                    home_team=home_team,
                    away_team=away_team,
                    season=self.current_season,
                    week=week or self.current_week
                )
                matchup_features.update(injury_features)
                
                logger.info(f"   🏥 Home injury impact: {injury_features.get('home_total_injury_impact', 0):.3f}")
                logger.info(f"   🏥 Away injury impact: {injury_features.get('away_total_injury_impact', 0):.3f}")
                
            except Exception as e:
                logger.warning(f"Could not add injury features, using defaults: {e}")
                # Add default injury features to maintain feature parity
                injury_features = self._get_default_injury_features()
                matchup_features.update(injury_features)
        else:
            # Always add injury features (as 0.0) to match training data
            injury_features = self._get_default_injury_features()
            matchup_features.update(injury_features)
        
        # Calculate injury-adjusted EPA (always, even with 0.0 impact)
        matchup_features['home_injury_adj_off_epa'] = (
            matchup_features['home_off_epa_per_play'] * 
            (1 - matchup_features.get('home_off_injury_impact', 0))
        )
        matchup_features['away_injury_adj_off_epa'] = (
            matchup_features['away_off_epa_per_play'] * 
            (1 - matchup_features.get('away_off_injury_impact', 0))
        )
        matchup_features['home_injury_adj_def_epa'] = (
            matchup_features['home_def_epa_per_play'] * 
            (1 + matchup_features.get('home_def_injury_impact', 0))
        )
        matchup_features['away_injury_adj_def_epa'] = (
            matchup_features['away_def_epa_per_play'] * 
            (1 + matchup_features.get('away_def_injury_impact', 0))
        )
        
        return matchup_features
    
    def create_default_features(self, spread: float, week: int, market_home_ml_prob: float = None, market_away_ml_prob: float = None) -> Dict:
        """Create default league-average features when team data is unavailable"""
        # Use league average values
        features = {
            'spread': spread,
            'home_wins': 3.5,
            'home_losses': 3.5,
            'home_win_pct': 0.5,
            'record_diff': 0.0,
            'home_off_epa_per_play': 0.0,
            'home_off_success_rate': 0.5,
            'home_off_explosive_play_rate': 0.15,
            'home_def_epa_per_play': 0.0,
            'home_def_success_rate': 0.5,
            'home_off_red_zone_plays': 15.0,
            'home_off_red_zone_td_rate': 0.55,
            'home_off_3rd_down_conv_rate': 0.40,
            'home_off_turnover_rate': 0.025,
            'home_def_turnover_rate': 0.025,
            'home_off_yards_per_play': 5.5,
            'home_def_yards_per_play': 5.5,
            'home_off_pass_rate': 0.60,
            'home_off_run_rate': 0.40,
            'home_avg_time_to_throw': 2.7,
            'home_avg_completed_air_yards': 6.5,
            'home_avg_air_yards_diff': 4.5,
            'home_pass_aggressiveness': 0.8,
            'home_avg_rush_yards_ngs': 4.0,
            'home_avg_time_to_los': 1.5,
            'home_rush_attempts': 25.0,
            'home_passing_yards_per_game': 220.0,
            'home_passing_tds_per_game': 1.5,
            'home_rushing_yards_per_game': 110.0,
            'home_rushing_tds_per_game': 1.0,
            'home_penalties_per_game': 6.0,
            'home_interceptions_per_game': 1.0,
            'home_sacks_taken_per_game': 2.5,
            'away_wins': 3.5,
            'away_losses': 3.5,
            'away_win_pct': 0.5,
            'away_off_epa_per_play': 0.0,
            'away_off_success_rate': 0.5,
            'away_off_explosive_play_rate': 0.15,
            'away_def_epa_per_play': 0.0,
            'away_def_success_rate': 0.5,
            'away_off_red_zone_plays': 15.0,
            'away_off_red_zone_td_rate': 0.55,
            'away_off_3rd_down_conv_rate': 0.40,
            'away_off_turnover_rate': 0.025,
            'away_def_turnover_rate': 0.025,
            'away_off_yards_per_play': 5.5,
            'away_def_yards_per_play': 5.5,
            'away_off_pass_rate': 0.60,
            'away_off_run_rate': 0.40,
            'away_avg_time_to_throw': 2.7,
            'away_avg_completed_air_yards': 6.5,
            'away_avg_air_yards_diff': 4.5,
            'away_pass_aggressiveness': 0.8,
            'away_avg_rush_yards_ngs': 4.0,
            'away_avg_time_to_los': 1.5,
            'away_rush_attempts': 25.0,
            'away_passing_yards_per_game': 220.0,
            'away_passing_tds_per_game': 1.5,
            'away_rushing_yards_per_game': 110.0,
            'away_rushing_tds_per_game': 1.0,
            'away_penalties_per_game': 6.0,
            'away_interceptions_per_game': 1.0,
            'away_sacks_taken_per_game': 2.5,
            'week': week,
        }
        
        # Add market odds features
        if market_home_ml_prob is not None and market_away_ml_prob is not None:
            total = market_home_ml_prob + market_away_ml_prob
            if total > 0:
                features['market_home_ml_prob'] = market_home_ml_prob / total
                features['market_away_ml_prob'] = market_away_ml_prob / total
                features['market_prob_diff'] = features['market_home_ml_prob'] - features['market_away_ml_prob']
            else:
                features['market_home_ml_prob'] = 0.5
                features['market_away_ml_prob'] = 0.5
                features['market_prob_diff'] = 0.0
        else:
            # Derive from spread if market odds not provided
            if spread is not None:
                home_ml, away_ml = self.spread_to_ml_odds(spread)
                home_implied = self.american_to_implied_prob(home_ml)
                away_implied = self.american_to_implied_prob(away_ml)
                if home_implied is not None and away_implied is not None:
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
        
        return features
    
    def load_and_calculate_features(self):
        """Main method to load data and calculate all team features"""
        logger.info(f"Loading {self.current_season} season data up to week {self.current_week}")
        
        # Fetch play-by-play data
        pbp_df = self.fetch_current_season_data()
        
        if pbp_df is None or len(pbp_df) == 0:
            logger.error("No play-by-play data available")
            return None
        
        # Calculate team features
        self.team_features = self.calculate_team_features(pbp_df)
        
        # Load injury data if enabled
        if self.include_injuries and self.injury_engine:
            try:
                logger.info("Loading injury data for current week...")
                self.injury_engine.load_all_data()
                logger.info("✅ Injury data loaded")
            except Exception as e:
                logger.warning(f"Could not load injury data: {e}")
                self.injury_engine = None
        
        logger.info(f"✅ Features ready for {len(self.team_features)} teams")
        
        return self.team_features
    
    def _get_default_injury_features(self) -> Dict:
        """Default injury features to maintain parity with training data"""
        return {
            'home_off_injury_impact': 0.0,
            'home_def_injury_impact': 0.0,
            'home_total_injury_impact': 0.0,
            'home_qb_out': 0,
            'home_top_wr_out': 0,
            'home_top_cb_out': 0,
            'home_lt_out': 0,
            'home_edge_out': 0,
            'home_multiple_ol_out': 0,
            'home_players_out': 0,
            'home_players_doubtful': 0,
            'home_players_questionable': 0,
            'home_total_injuries': 0,
            'home_premium_injuries': 0,
            'away_off_injury_impact': 0.0,
            'away_def_injury_impact': 0.0,
            'away_total_injury_impact': 0.0,
            'away_qb_out': 0,
            'away_top_wr_out': 0,
            'away_top_cb_out': 0,
            'away_lt_out': 0,
            'away_edge_out': 0,
            'away_multiple_ol_out': 0,
            'away_players_out': 0,
            'away_players_doubtful': 0,
            'away_players_questionable': 0,
            'away_total_injuries': 0,
            'away_premium_injuries': 0,
            'injury_impact_differential': 0.0,
            'combined_injury_impact': 0.0,
        }


if __name__ == "__main__":
    # Test the feature fetcher
    fetcher = CurrentSeasonFeatures(current_season=2025, current_week=7)
    features = fetcher.load_and_calculate_features()
    
    if features is not None:
        print("\n📊 Team Features Summary:")
        print(features[['team', 'off_epa_play', 'def_epa_play', 'plays']].head(10))
        
        # Test matchup features
        print("\n🏈 Example Matchup: Vikings @ Chargers")
        matchup = fetcher.create_matchup_features('LAC', 'MIN', -6.8)
        print(f"Home Off EPA: {matchup['home_off_epa']:.3f}")
        print(f"Away Off EPA: {matchup['away_off_epa']:.3f}")
        print(f"Total EPA Edge: {matchup['total_epa_edge']:.3f}")

