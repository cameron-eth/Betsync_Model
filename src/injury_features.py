"""
Injury Feature Engineering

Uses nflreadpy to load injury data and calculate injury impact features:
- Team injury impact scores
- Key player out flags
- Injury-adjusted performance metrics
- Historical injury context
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

try:
    import nflreadpy as nfl
except ImportError:
    nfl = None

try:
    from .injury_impact_weights import (
        get_injury_impact,
        categorize_injury_severity,
        KEY_POSITION_FLAGS,
        PREMIUM_POSITIONS
    )
    from .utils import logger, print_section_header
except ImportError:
    from injury_impact_weights import (
        get_injury_impact,
        categorize_injury_severity,
        KEY_POSITION_FLAGS,
        PREMIUM_POSITIONS
    )
    from utils import logger, print_section_header

logger = logging.getLogger(__name__)


class InjuryFeatureEngine:
    """Extract injury features from nflreadpy data"""
    
    def __init__(self, seasons: List[int] = None):
        """Initialize with seasons to load"""
        if seasons is None:
            # Injury data available since 2009
            seasons = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
        self.seasons = seasons
        self.injuries_df = None
        self.depth_charts_df = None
        self.rosters_weekly_df = None
        self.snap_counts_df = None
        
    def load_all_data(self):
        """Load all injury-related data from nflreadpy"""
        print_section_header("Loading Injury Data from nflreadpy")
        
        if nfl is None:
            raise ImportError("nflreadpy not installed. Run: pip install nflreadpy")
        
        print(f"📊 Loading injury data for seasons: {self.seasons}")
        
        # Load injury reports (HYBRID APPROACH)
        print("\n1️⃣ Loading Injury Reports...")
        try:
            injuries = nfl.load_injuries(self.seasons)
            self.injuries_df = injuries.to_pandas()
            print(f"   ✅ Loaded {len(self.injuries_df):,} injury reports")
        except Exception as e:
            # For current season (2025), injury data may not be available yet
            # Fall back to using ONLY rosters_weekly which has current injury status
            logger.warning(f"Injury reports not available: {e}")
            print(f"   ⚠️ Injury reports not available, will use weekly rosters instead")
            self.injuries_df = pd.DataFrame()
        
        # Load depth charts (to identify starters)
        print("\n2️⃣ Loading Depth Charts...")
        try:
            depth_charts = nfl.load_depth_charts(self.seasons)
            self.depth_charts_df = depth_charts.to_pandas()
            print(f"   ✅ Loaded {len(self.depth_charts_df):,} depth chart entries")
        except Exception as e:
            logger.warning(f"Could not load depth charts: {e}")
            self.depth_charts_df = pd.DataFrame()
        
        # Load weekly rosters (injury status by week)
        print("\n3️⃣ Loading Weekly Rosters (Injury Status)...")
        try:
            rosters_weekly = nfl.load_rosters_weekly(self.seasons)
            self.rosters_weekly_df = rosters_weekly.to_pandas()
            print(f"   ✅ Loaded {len(self.rosters_weekly_df):,} weekly roster records")
        except Exception as e:
            logger.warning(f"Could not load weekly rosters: {e}")
            self.rosters_weekly_df = pd.DataFrame()
        
        # Load snap counts (to weight player importance)
        print("\n4️⃣ Loading Snap Counts...")
        try:
            snap_counts = nfl.load_snap_counts(self.seasons)
            self.snap_counts_df = snap_counts.to_pandas()
            print(f"   ✅ Loaded {len(self.snap_counts_df):,} snap count records")
        except Exception as e:
            logger.warning(f"Could not load snap counts: {e}")
            self.snap_counts_df = pd.DataFrame()
        
        print("\n✅ All injury data loaded successfully!\n")
    
    def get_player_depth_rank(
        self, 
        team: str, 
        player: str, 
        position: str, 
        season: int, 
        week: int
    ) -> str:
        """
        Get player's depth chart ranking (1 = starter, 2 = backup, etc.)
        """
        if self.depth_charts_df is None or len(self.depth_charts_df) == 0:
            return '1'  # Default to starter if no depth chart data
        
        team_col = self._get_team_column(self.depth_charts_df)
        if not team_col:
            return '1'
        
        # Filter depth chart for team, season, week
        depth_data = self.depth_charts_df[
            (self.depth_charts_df[team_col] == team) &
            (self.depth_charts_df['season'] == season) &
            (self.depth_charts_df['week'] == week)
        ]
        
        if len(depth_data) == 0:
            return '1'
        
        # Find player's rank at their position
        player_depth = pd.DataFrame()
        if 'full_name' in depth_data.columns:
            player_depth = depth_data[
                depth_data['full_name'].str.contains(player, case=False, na=False)
            ]
        
        if len(player_depth) == 0 and 'gsis_id' in depth_data.columns:
            player_depth = depth_data[
                depth_data['gsis_id'].str.contains(player, case=False, na=False)
            ]
        
        if len(player_depth) > 0:
            # Get depth chart rank (typically 1, 2, 3, etc.)
            rank = player_depth.iloc[0].get('depth_team', 1)
            return str(int(rank))
        
        return '1'  # Default to starter
    
    def get_player_snap_percentage(
        self, 
        team: str, 
        player: str, 
        season: int, 
        week: int
    ) -> Optional[float]:
        """
        Get player's average snap percentage (used to weight importance)
        """
        if self.snap_counts_df is None or len(self.snap_counts_df) == 0:
            return None
        
        team_col = self._get_team_column(self.snap_counts_df)
        if not team_col:
            return None
        
        # Get snap counts for previous 4 weeks
        snap_data = self.snap_counts_df[
            (self.snap_counts_df[team_col] == team) &
            (self.snap_counts_df['season'] == season) &
            (self.snap_counts_df['week'] < week) &
            (self.snap_counts_df['week'] >= week - 4)
        ]
        
        # Filter by player name
        if 'player' in snap_data.columns:
            snap_data = snap_data[
                snap_data['player'].str.contains(player, case=False, na=False)
            ]
        
        if len(snap_data) > 0:
            # Return average snap percentage
            return snap_data['offense_pct'].mean() if 'offense_pct' in snap_data.columns else None
        
        return None
    
    def _get_team_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the team column name in a dataframe"""
        possible_names = ['team', 'team_abbr', 'team_abb', 'club_code']
        for name in possible_names:
            if name in df.columns:
                return name
        return None
    
    def get_team_injuries(
        self, 
        team: str, 
        season: int, 
        week: int
    ) -> pd.DataFrame:
        """
        Get all injuries for a team in a specific week.
        Uses multiple data sources for comprehensive injury tracking.
        
        HYBRID APPROACH:
        - Use nflreadpy load_injuries() when available (2009-2024)
        - For 2025+: No injury data yet (nflverse publishes with delay)
        """
        injuries = []
        
        # Source 1: Injury reports from nflreadpy
        if self.injuries_df is not None and len(self.injuries_df) > 0:
            team_col = self._get_team_column(self.injuries_df)
            if team_col:
                injury_reports = self.injuries_df[
                    (self.injuries_df[team_col] == team) &
                    (self.injuries_df['season'] == season) &
                    (self.injuries_df['week'] == week)
                ].copy()
                
                # Standardize column names
                if len(injury_reports) > 0:
                    if 'full_name' in injury_reports.columns and 'player_name' not in injury_reports.columns:
                        injury_reports['player_name'] = injury_reports['full_name']
                    if 'player' in injury_reports.columns and 'player_name' not in injury_reports.columns:
                        injury_reports['player_name'] = injury_reports['player']
                        
                    injuries.append(injury_reports)
        
        # Source 2: Weekly rosters with IR/Reserve status (supplemental)
        # Note: rosters_weekly has roster statuses (ACT/RES/INA), not injury reports
        # We only use this to identify players on IR/PUP lists
        if self.rosters_weekly_df is not None and len(self.rosters_weekly_df) > 0 and season < 2025:
            team_col = self._get_team_column(self.rosters_weekly_df)
            if team_col:
                roster_status = self.rosters_weekly_df[
                    (self.rosters_weekly_df[team_col] == team) &
                    (self.rosters_weekly_df['season'] == season) &
                    (self.rosters_weekly_df['week'] == week) &
                    (self.rosters_weekly_df['status'].isin(['RES']))  # Reserve = IR/PUP
                ].copy()
                
                if len(roster_status) > 0:
                    # These are IR/PUP players
                    roster_status['status'] = 'IR'  # Mark as Out
                    if 'full_name' in roster_status.columns and 'player_name' not in roster_status.columns:
                        roster_status['player_name'] = roster_status['full_name']
                    
                    injuries.append(roster_status)
        
        # Combine all sources
        if injuries:
            combined = pd.concat(injuries, ignore_index=True)
            
            # Remove duplicates (same player from multiple sources)
            # Use columns that exist in the combined dataframe
            dedup_cols = []
            if 'player_name' in combined.columns:
                dedup_cols.append('player_name')
            if 'position' in combined.columns:
                dedup_cols.append('position')
            
            if dedup_cols:
                combined = combined.drop_duplicates(subset=dedup_cols, keep='first')
            
            return combined
        
        return pd.DataFrame()
    
    def calculate_injury_impact(
        self, 
        team: str, 
        season: int, 
        week: int
    ) -> Dict:
        """
        Calculate comprehensive injury impact features for a team.
        
        Returns dict with:
        - Total injury impact scores (offense, defense)
        - Key player out flags
        - Injury counts by severity
        - Premium position injuries
        """
        injuries = self.get_team_injuries(team, season, week)
        
        if len(injuries) == 0:
            return self._get_default_injury_features()
        
        features = {}
        
        # Track impacts
        off_impact = 0.0
        def_impact = 0.0
        
        # Track key position flags
        qb_out = 0
        top_wr_out = 0
        top_cb_out = 0
        lt_out = 0
        edge_out = 0
        ol_count = 0
        
        # Track injury counts by severity
        out_count = 0
        doubtful_count = 0
        questionable_count = 0
        
        # Track premium position injuries
        premium_injuries = 0
        
        # Process each injury
        for _, injury in injuries.iterrows():
            # Handle different column naming conventions
            player_name = injury.get('player_name')
            if pd.isna(player_name) or not player_name:
                player_name = injury.get('full_name', injury.get('player', 'Unknown'))
            
            position = injury.get('position', 'UNK')
            if pd.isna(position) or not position:
                position = 'UNK'
            
            injury_status = injury.get('report_status')
            if pd.isna(injury_status) or not injury_status:
                injury_status = injury.get('status', 'Questionable')
            
            # Get depth chart rank
            depth_rank = self.get_player_depth_rank(team, player_name, position, season, week)
            
            # Get snap percentage
            snap_pct = self.get_player_snap_percentage(team, player_name, season, week)
            
            # Calculate impact
            impact = get_injury_impact(position, depth_rank, injury_status, snap_pct)
            
            # Categorize by offense/defense
            offensive_positions = ['QB', 'RB', 'WR', 'TE', 'FB', 'LT', 'LG', 'C', 'RG', 'RT', 'OL']
            if position in offensive_positions:
                off_impact += impact
            else:
                def_impact += impact
            
            # Key position flags
            if position == 'QB' and depth_rank == '1' and injury_status in ['Out', 'Doubtful']:
                qb_out = 1
            
            if position == 'WR' and depth_rank == '1' and injury_status in ['Out', 'Doubtful']:
                top_wr_out = 1
            
            if position == 'CB' and depth_rank == '1' and injury_status in ['Out', 'Doubtful']:
                top_cb_out = 1
            
            if position == 'LT' and depth_rank == '1' and injury_status in ['Out', 'Doubtful']:
                lt_out = 1
            
            if position in ['EDGE', 'DE'] and depth_rank == '1' and injury_status in ['Out', 'Doubtful']:
                edge_out = 1
            
            if position in ['LT', 'LG', 'C', 'RG', 'RT', 'OL'] and injury_status in ['Out', 'Doubtful']:
                ol_count += 1
            
            # Count by severity
            if injury_status == 'Out':
                out_count += 1
            elif injury_status == 'Doubtful':
                doubtful_count += 1
            elif injury_status == 'Questionable':
                questionable_count += 1
            
            # Premium positions
            if position in PREMIUM_POSITIONS and depth_rank == '1':
                premium_injuries += 1
        
        # Compile features
        features['off_injury_impact'] = min(off_impact, 1.0)
        features['def_injury_impact'] = min(def_impact, 1.0)
        features['total_injury_impact'] = min(off_impact + def_impact, 1.0)
        
        # Key player flags
        features['qb_out'] = qb_out
        features['top_wr_out'] = top_wr_out
        features['top_cb_out'] = top_cb_out
        features['lt_out'] = lt_out
        features['edge_out'] = edge_out
        features['multiple_ol_out'] = 1 if ol_count >= 2 else 0
        
        # Injury counts
        features['players_out'] = out_count
        features['players_doubtful'] = doubtful_count
        features['players_questionable'] = questionable_count
        features['total_injuries'] = len(injuries)
        
        # Premium position injuries
        features['premium_injuries'] = premium_injuries
        
        # Severity category
        severity = categorize_injury_severity(features['total_injury_impact'])
        features['injury_severity'] = severity
        
        return features
    
    def _get_default_injury_features(self) -> Dict:
        """Default injury features when no injury data available"""
        return {
            'off_injury_impact': 0.0,
            'def_injury_impact': 0.0,
            'total_injury_impact': 0.0,
            'qb_out': 0,
            'top_wr_out': 0,
            'top_cb_out': 0,
            'lt_out': 0,
            'edge_out': 0,
            'multiple_ol_out': 0,
            'players_out': 0,
            'players_doubtful': 0,
            'players_questionable': 0,
            'total_injuries': 0,
            'premium_injuries': 0,
            'injury_severity': 'MINIMAL',
        }
    
    def get_injury_features_for_game(
        self, 
        home_team: str, 
        away_team: str, 
        season: int, 
        week: int
    ) -> Dict:
        """
        Get all injury features for a game (both teams).
        
        Returns dict with home_* and away_* prefixed features.
        """
        print(f"🏥 Extracting injury features: {away_team} @ {home_team} (Week {week}, {season})")
        
        # Home team injuries
        home_injuries = self.calculate_injury_impact(home_team, season, week)
        
        # Away team injuries
        away_injuries = self.calculate_injury_impact(away_team, season, week)
        
        # Combine with prefixes
        all_features = {}
        
        for key, val in home_injuries.items():
            if key != 'injury_severity':  # Don't include text categories
                all_features[f'home_{key}'] = val
        
        for key, val in away_injuries.items():
            if key != 'injury_severity':
                all_features[f'away_{key}'] = val
        
        # Add comparative features
        all_features['injury_impact_differential'] = (
            home_injuries['total_injury_impact'] - away_injuries['total_injury_impact']
        )
        all_features['combined_injury_impact'] = (
            home_injuries['total_injury_impact'] + away_injuries['total_injury_impact']
        )
        
        print(f"   ✅ Home injuries: {home_injuries['total_injury_impact']:.3f} impact")
        print(f"   ✅ Away injuries: {away_injuries['total_injury_impact']:.3f} impact")
        
        return all_features


def main():
    """Test injury feature extraction"""
    print_section_header("Testing Injury Feature Engine")
    
    engine = InjuryFeatureEngine(seasons=[2023, 2024])
    engine.load_all_data()
    
    # Test on a sample game
    features = engine.get_injury_features_for_game(
        home_team='KC',
        away_team='BUF',
        season=2024,
        week=7
    )
    
    print(f"\n📊 Sample Injury Features:")
    for key, val in features.items():
        print(f"   {key}: {val}")


if __name__ == "__main__":
    main()

