"""
NBA Current Season Feature Fetcher
Uses nba_api to fetch current season stats up to current date
"""

import pandas as pd
from datetime import datetime
from typing import Dict, Optional
import logging
import time

try:
    from nba_api.stats.endpoints import (
        teamgamelog, leaguegamelog, leaguedashteamstats
    )
except ImportError:
    pass

try:
    from .nba_advanced_features import NBAAdvancedFeatureEngine
    from .nba_utils import logger
except ImportError:
    from nba_advanced_features import NBAAdvancedFeatureEngine
    from nba_utils import logger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NBACurrentSeasonFeatures:
    """Fetch and calculate current season features for predictions"""
    
    def __init__(self, current_season: int = 2025, include_injuries: bool = True):
        self.current_season = current_season
        self.include_injuries = include_injuries
        self.team_features = None
        self.feature_engine = None
        
        # Initialize feature engine
        self.feature_engine = NBAAdvancedFeatureEngine(seasons=[current_season])
        
    def fetch_current_season_data(self):
        """Fetch current season data up to today"""
        logger.info(f"Fetching {self.current_season} season data...")
        
        try:
            # Load feature engine data
            self.feature_engine.load_all_data()
            
            logger.info(f"Loaded current season data")
            return True
            
        except Exception as e:
            logger.error(f"Error fetching current season data: {e}")
            return False
    
    def load_and_calculate_features(self) -> Optional[pd.DataFrame]:
        """Load and calculate team features for current season"""
        logger.info("Calculating current season team features...")
        
        if not self.fetch_current_season_data():
            return None
        
        # Get all teams
        from .nba_config import NBA_TEAMS
        
        team_features_list = []
        today = datetime.now().strftime('%Y-%m-%d')
        
        for team_abbr in NBA_TEAMS.keys():
            try:
                features = self.feature_engine.calculate_team_features(
                    team_abbr, self.current_season, today
                )
                features['team'] = team_abbr
                team_features_list.append(features)
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"Could not calculate features for {team_abbr}: {e}")
                continue
        
        if team_features_list:
            self.team_features = pd.DataFrame(team_features_list)
            logger.info(f"Calculated features for {len(self.team_features)} teams")
            return self.team_features
        else:
            return None

