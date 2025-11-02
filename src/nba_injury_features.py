"""
NBA Injury Feature Engineering

Uses nba_api and other sources to load injury data and calculate injury impact features:
- Player injury status
- Key player out flags (stars, starters)
- Injury-adjusted performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

try:
    from nba_api.stats.endpoints import (
        commonplayerinfo, teamroster, boxscoretraditionalv2
    )
    from nba_api.stats.staticdata import players
except ImportError:
    pass

try:
    from .nba_config import NBA_TEAMS, NBA_TEAMS_REVERSE
    from .nba_utils import logger, print_section_header, normalize_team_name
except ImportError:
    from nba_config import NBA_TEAMS, NBA_TEAMS_REVERSE
    from nba_utils import logger, print_section_header, normalize_team_name

logger = logging.getLogger(__name__)


class NBAInjuryFeatureEngine:
    """Extract injury features for NBA games"""
    
    def __init__(self, seasons: List[int] = None):
        """Initialize with seasons to load"""
        if seasons is None:
            seasons = list(range(2015, 2025))
        self.seasons = seasons
        self.injury_data = {}  # Will store injury data by team/date
    
    def load_all_data(self):
        """Load injury data (placeholder - NBA injury data is less structured than NFL)"""
        print_section_header("Loading NBA Injury Data")
        
        print("⚠️  NBA injury data is less structured than NFL")
        print("   Will use team rosters and box scores to infer player availability")
        print("   This is a simplified approach - full injury reports require external sources")
        
        # For now, we'll use a simplified approach
        # In production, you might want to integrate with:
        # - ESPN API
        # - Rotowire
        # - Other injury tracking services
        
        print("\n✅ Injury engine initialized (will infer from box scores)")
    
    def get_game_injury_features(
        self, 
        home_team_abbr: str, 
        away_team_abbr: str, 
        game_date: str
    ) -> Dict:
        """
        Get injury features for a game
        
        Returns:
            Dictionary of injury features
        """
        features = {}
        
        # For now, return default features
        # In production, this would check actual injury reports
        
        # Default: no major injuries
        features['home_star_player_out'] = 0
        features['away_star_player_out'] = 0
        features['home_starters_out'] = 0
        features['away_starters_out'] = 0
        features['home_injury_impact'] = 0.0
        features['away_injury_impact'] = 0.0
        
        return features
    
    def _get_star_players(self, team_abbr: str, season: int) -> List[int]:
        """
        Get list of star player IDs for a team
        This is a simplified approach - in production, you'd maintain a curated list
        """
        # Placeholder - would maintain a list of star players
        # Could be based on:
        # - All-Star selections
        # - Minutes played
        # - Usage rate
        # - PER or other advanced metrics
        
        return []
    
    def _check_player_availability(
        self, 
        team_abbr: str, 
        player_id: int, 
        game_date: str
    ) -> bool:
        """
        Check if a player was available for a game
        Returns True if player played, False if injured/rested
        """
        # Would check box scores to see if player played
        # If player is in box score, they were available
        return True  # Default assumption

