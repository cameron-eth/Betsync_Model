"""
NBA-specific utility functions for BetSync NBA Prediction Model
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

try:
    from .nba_config import (
        SUPABASE_URL, SUPABASE_KEY, NBA_TEAMS, NBA_TEAMS_REVERSE,
        NBA_TABLE_TEAM_STATS, NBA_TABLE_PLAYER_STATS, LOG_LEVEL
    )
except ImportError:
    from nba_config import (
        SUPABASE_URL, SUPABASE_KEY, NBA_TEAMS, NBA_TEAMS_REVERSE,
        NBA_TABLE_TEAM_STATS, NBA_TABLE_PLAYER_STATS, LOG_LEVEL
    )

from supabase import create_client, Client

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
    Normalize NBA team name to full official name
    
    Args:
        team_name: Team name or abbreviation
        
    Returns:
        Full official team name
    """
    team_name = team_name.strip()
    
    # If it's an abbreviation, convert to full name
    if team_name in NBA_TEAMS:
        return NBA_TEAMS[team_name]
    
    # If it's already a full name, return as is
    if team_name in NBA_TEAMS_REVERSE:
        return team_name
    
    # Try case-insensitive match
    for abbr, full_name in NBA_TEAMS.items():
        if team_name.lower() == full_name.lower():
            return full_name
        if team_name.upper() == abbr.upper():
            return full_name
    
    # Handle common variations
    variations = {
        'Brooklyn': 'Brooklyn Nets',
        'LA Clippers': 'LA Clippers',
        'LA Lakers': 'Los Angeles Lakers',
        'Los Angeles Clippers': 'LA Clippers',
        'New Orleans': 'New Orleans Pelicans',
        'Oklahoma City': 'Oklahoma City Thunder',
        'Philadelphia': 'Philadelphia 76ers',
        'San Antonio': 'San Antonio Spurs',
        'Golden State': 'Golden State Warriors',
        'Portland': 'Portland Trail Blazers',
        'Phoenix': 'Phoenix Suns',
        'Sacramento': 'Sacramento Kings',
        'Utah': 'Utah Jazz',
        'Washington': 'Washington Wizards',
        'Charlotte': 'Charlotte Hornets',
        'Miami': 'Miami Heat',
        'Orlando': 'Orlando Magic',
        'Atlanta': 'Atlanta Hawks',
        'Boston': 'Boston Celtics',
        'Chicago': 'Chicago Bulls',
        'Cleveland': 'Cleveland Cavaliers',
        'Detroit': 'Detroit Pistons',
        'Indiana': 'Indiana Pacers',
        'Milwaukee': 'Milwaukee Bucks',
        'New York': 'New York Knicks',
        'Toronto': 'Toronto Raptors',
        'Dallas': 'Dallas Mavericks',
        'Denver': 'Denver Nuggets',
        'Houston': 'Houston Rockets',
        'Memphis': 'Memphis Grizzlies',
        'Minnesota': 'Minnesota Timberwolves',
    }
    
    if team_name in variations:
        return variations[team_name]
    
    logger.warning(f"Unknown NBA team name: {team_name}")
    return team_name


def get_team_abbr(team_name: str) -> Optional[str]:
    """
    Get NBA team abbreviation from full name
    
    Args:
        team_name: Full team name
        
    Returns:
        Team abbreviation or None
    """
    normalized = normalize_team_name(team_name)
    return NBA_TEAMS_REVERSE.get(normalized)


def calculate_win_percentage(wins: int, losses: int) -> float:
    """
    Calculate win percentage for NBA (no ties)
    
    Args:
        wins: Number of wins
        losses: Number of losses
        
    Returns:
        Win percentage (0.0 to 1.0)
    """
    total_games = wins + losses
    if total_games == 0:
        return 0.0
    return wins / total_games


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
    """Safely convert value to float"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value, default=0) -> int:
    """Safely convert value to int"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def get_season_from_date(date: str) -> int:
    """
    Get NBA season year from date
    (Season runs Oct-Apr, season year is year of end date)
    
    Args:
        date: Date string (YYYY-MM-DD)
        
    Returns:
        Season year (e.g., 2024 for 2023-24 season)
    """
    try:
        dt = pd.to_datetime(date)
        # If before October, it's previous year's season
        if dt.month < 10:
            return dt.year
        return dt.year + 1
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


def calculate_offensive_rating(points: float, possessions: float) -> float:
    """
    Calculate offensive rating (points per 100 possessions)
    
    Args:
        points: Total points scored
        possessions: Total possessions
        
    Returns:
        Offensive rating
    """
    if possessions == 0:
        return 0.0
    return (points / possessions) * 100


def calculate_defensive_rating(points_allowed: float, possessions: float) -> float:
    """
    Calculate defensive rating (points allowed per 100 possessions)
    
    Args:
        points_allowed: Total points allowed
        possessions: Total possessions
        
    Returns:
        Defensive rating
    """
    if possessions == 0:
        return 0.0
    return (points_allowed / possessions) * 100


def calculate_net_rating(off_rating: float, def_rating: float) -> float:
    """
    Calculate net rating (offensive rating - defensive rating)
    
    Args:
        off_rating: Offensive rating
        def_rating: Defensive rating
        
    Returns:
        Net rating
    """
    return off_rating - def_rating


def calculate_true_shooting_percentage(points: float, fga: float, fta: float) -> float:
    """
    Calculate True Shooting Percentage
    TS% = Points / (2 * (FGA + 0.44 * FTA))
    
    Args:
        points: Total points
        fga: Field goal attempts
        fta: Free throw attempts
        
    Returns:
        True shooting percentage (0.0 to 1.0)
    """
    denominator = 2 * (fga + 0.44 * fta)
    if denominator == 0:
        return 0.0
    return points / denominator


def calculate_effective_fg_percentage(fgm: float, fg3m: float, fga: float) -> float:
    """
    Calculate Effective Field Goal Percentage
    eFG% = (FGM + 0.5 * 3PM) / FGA
    
    Args:
        fgm: Field goals made
        fg3m: 3-point field goals made
        fga: Field goal attempts
        
    Returns:
        Effective field goal percentage (0.0 to 1.0)
    """
    if fga == 0:
        return 0.0
    return (fgm + 0.5 * fg3m) / fga


def calculate_pace(possessions: float, minutes: float) -> float:
    """
    Calculate pace (possessions per 48 minutes)
    
    Args:
        possessions: Total possessions
        minutes: Total minutes played
        
    Returns:
        Pace (possessions per 48 minutes)
    """
    if minutes == 0:
        return 0.0
    return (possessions / minutes) * 48


def print_separator(char="=", length=80):
    """Print a separator line"""
    print(char * length)


def print_section_header(title: str):
    """Print a formatted section header"""
    print_separator()
    print(f"🏀 {title}")
    print_separator()


def fetch_team_season_stats(
    team_name: str,
    season: int
) -> Optional[Dict]:
    """
    Fetch NBA team season statistics from Supabase
    
    Args:
        team_name: Team name
        season: Season year
        
    Returns:
        Dictionary of team stats or None
    """
    try:
        supabase = get_supabase_client()
        result = supabase.table(NBA_TABLE_TEAM_STATS).select("*").eq("team_name", team_name).eq("season", season).execute()
        
        if result.data and len(result.data) > 0:
            return result.data[0]
        return None
    except Exception as e:
        logger.error(f"Error fetching NBA team stats: {e}")
        return None

