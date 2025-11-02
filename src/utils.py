"""
Utility functions for BetSync NFL Prediction Model
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from supabase import create_client, Client

from .config import (
    SUPABASE_URL, SUPABASE_KEY, NFL_TEAMS, NFL_TEAMS_REVERSE,
    TABLE_TEAM_STATS, TABLE_PLAYER_STATS, LOG_LEVEL
)

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_supabase_client() -> Client:
    """Create and return Supabase client"""
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def normalize_team_name(team_name: str) -> str:
    """
    Normalize team name to full official name
    
    Args:
        team_name: Team name or abbreviation
        
    Returns:
        Full official team name
    """
    team_name = team_name.strip()
    
    # If it's an abbreviation, convert to full name
    if team_name in NFL_TEAMS:
        return NFL_TEAMS[team_name]
    
    # If it's already a full name, return as is
    if team_name in NFL_TEAMS_REVERSE:
        return team_name
    
    # Try case-insensitive match
    for abbr, full_name in NFL_TEAMS.items():
        if team_name.lower() == full_name.lower():
            return full_name
        if team_name.upper() == abbr.upper():
            return full_name
    
    # Handle legacy team names
    legacy_names = {
        'Washington Football Team': 'Washington Commanders',
        'Washington Redskins': 'Washington Commanders',
        'Oakland Raiders': 'Las Vegas Raiders',
        'San Diego Chargers': 'Los Angeles Chargers',
        'St. Louis Rams': 'Los Angeles Rams'
    }
    
    if team_name in legacy_names:
        return legacy_names[team_name]
    
    logger.warning(f"Unknown team name: {team_name}")
    return team_name


def get_team_abbr(team_name: str) -> Optional[str]:
    """
    Get team abbreviation from full name
    
    Args:
        team_name: Full team name
        
    Returns:
        Team abbreviation or None
    """
    normalized = normalize_team_name(team_name)
    return NFL_TEAMS_REVERSE.get(normalized)


def calculate_win_percentage(wins: int, losses: int, ties: int = 0) -> float:
    """
    Calculate win percentage (ties count as 0.5 wins)
    
    Args:
        wins: Number of wins
        losses: Number of losses
        ties: Number of ties
        
    Returns:
        Win percentage (0.0 to 1.0)
    """
    total_games = wins + losses + ties
    if total_games == 0:
        return 0.0
    return (wins + 0.5 * ties) / total_games


def calculate_days_between(date1: str, date2: str) -> int:
    """
    Calculate days between two dates
    
    Args:
        date1: First date (YYYY-MM-DD)
        date2: Second date (YYYY-MM-DD)
        
    Returns:
        Number of days between dates
    """
    try:
        d1 = pd.to_datetime(date1)
        d2 = pd.to_datetime(date2)
        return abs((d2 - d1).days)
    except:
        return 0


def safe_float(value, default=0.0) -> float:
    """
    Safely convert value to float
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Float value or default
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value, default=0) -> int:
    """
    Safely convert value to int
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Int value or default
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def get_season_from_date(date: str) -> int:
    """
    Get NFL season year from date
    (Season starts in September)
    
    Args:
        date: Date string (YYYY-MM-DD)
        
    Returns:
        Season year
    """
    try:
        dt = pd.to_datetime(date)
        # If before September, it's previous year's season
        if dt.month < 9:
            return dt.year - 1
        return dt.year
    except:
        return datetime.now().year


def calculate_confidence_score(
    win_probability: float,
    feature_importance: Dict[str, float],
    data_completeness: float
) -> float:
    """
    Calculate confidence score for prediction
    
    Args:
        win_probability: Predicted win probability (0.0 to 1.0)
        feature_importance: Dictionary of feature importances
        data_completeness: Percentage of features available (0.0 to 1.0)
        
    Returns:
        Confidence score (0.0 to 1.0)
    """
    # Higher confidence when prediction is more decisive (away from 0.5)
    decisiveness = abs(win_probability - 0.5) * 2
    
    # Weight by data completeness
    confidence = (decisiveness * 0.6 + data_completeness * 0.4)
    
    return min(max(confidence, 0.0), 1.0)


def get_confidence_label(confidence: float) -> str:
    """
    Get confidence label from score
    
    Args:
        confidence: Confidence score (0.0 to 1.0)
        
    Returns:
        Confidence label
    """
    if confidence >= 0.75:
        return "HIGH"
    elif confidence >= 0.60:
        return "MEDIUM"
    elif confidence >= 0.50:
        return "LOW"
    else:
        return "VERY_LOW"


def fetch_team_recent_games(
    team_name: str,
    n_games: int = 5,
    before_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch recent games for a team from Supabase
    
    Args:
        team_name: Team name
        n_games: Number of recent games
        before_date: Only get games before this date
        
    Returns:
        DataFrame of recent games
    """
    # This is a placeholder - actual implementation would query Supabase
    # For now, return empty DataFrame
    logger.info(f"Fetching {n_games} recent games for {team_name}")
    return pd.DataFrame()


def fetch_team_season_stats(
    team_name: str,
    season: int
) -> Optional[Dict]:
    """
    Fetch team season statistics from Supabase
    
    Args:
        team_name: Team name
        season: Season year
        
    Returns:
        Dictionary of team stats or None
    """
    try:
        supabase = get_supabase_client()
        result = supabase.table(TABLE_TEAM_STATS).select("*").eq("team_name", team_name).eq("season", season).execute()
        
        if result.data and len(result.data) > 0:
            return result.data[0]
        return None
    except Exception as e:
        logger.error(f"Error fetching team stats: {e}")
        return None


def print_separator(char="=", length=80):
    """Print a separator line"""
    print(char * length)


def print_section_header(title: str):
    """Print a formatted section header"""
    print_separator()
    print(f"🏈 {title}")
    print_separator()

