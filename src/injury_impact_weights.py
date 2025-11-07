"""
Injury Impact Weights - Position Importance & Status Multipliers

Defines the impact of injuries based on:
1. Player position (QB > WR1 > LT > etc.)
2. Injury status (Out > Doubtful > Questionable)
3. Depth chart ranking (Starter > Backup)
"""

from typing import Dict

# Position Impact Weights (0.0 - 1.0 scale)
# These represent the percentage impact on team performance when a player is out
POSITION_WEIGHTS: Dict[str, float] = {
    # Offense - Skill Positions
    'QB': 0.35,      # Quarterback - Massive impact on offense
    'WR': 0.08,      # Wide Receiver 1 (adjusted by depth chart)
    'RB': 0.06,      # Running Back
    'TE': 0.05,      # Tight End
    'FB': 0.02,      # Fullback
    
    # Offense - Offensive Line
    'LT': 0.12,      # Left Tackle - Protects QB's blind side
    'LG': 0.06,      # Left Guard
    'C': 0.07,       # Center - Calls protections
    'RG': 0.05,      # Right Guard
    'RT': 0.08,      # Right Tackle
    'OL': 0.07,      # Generic O-Line (if position not specified)
    
    # Defense - Front 7
    'EDGE': 0.09,    # Edge Rusher / DE - Pass rush impact
    'DE': 0.08,      # Defensive End
    'DT': 0.06,      # Defensive Tackle
    'NT': 0.05,      # Nose Tackle
    'ILB': 0.06,     # Inside Linebacker
    'MLB': 0.07,     # Middle Linebacker
    'OLB': 0.07,     # Outside Linebacker
    'LB': 0.06,      # Generic Linebacker
    
    # Defense - Secondary
    'CB': 0.08,      # Cornerback (adjusted by CB1 vs CB2)
    'S': 0.06,       # Safety
    'FS': 0.06,      # Free Safety
    'SS': 0.06,      # Strong Safety
    'DB': 0.06,      # Generic Defensive Back
    
    # Special Teams
    'K': 0.03,       # Kicker
    'P': 0.02,       # Punter
    'LS': 0.01,      # Long Snapper
}

# Depth Chart Multipliers
# Multiplies the position weight based on starter vs backup
DEPTH_MULTIPLIERS: Dict[str, float] = {
    '1': 1.0,        # Starter (QB1, WR1, CB1, etc.)
    '2': 0.5,        # Second string
    '3': 0.25,       # Third string
    '4': 0.10,       # Deep backup
}

# Special depth multipliers for key positions
DEPTH_POSITION_ADJUSTMENTS: Dict[str, Dict[str, float]] = {
    'QB': {
        '1': 1.0,    # Starting QB out = massive impact
        '2': 0.7,    # Backup QB still significant
        '3': 0.4,    # Third string QB
    },
    'WR': {
        '1': 1.0,    # WR1 - Elite target
        '2': 0.6,    # WR2 - Still important
        '3': 0.3,    # WR3 - Rotational
        '4': 0.1,    # WR4+ - Minimal impact
    },
    'CB': {
        '1': 1.0,    # CB1 - Covers best WR
        '2': 0.7,    # CB2 - Still critical
        '3': 0.4,    # CB3 - Nickel role
    },
    'EDGE': {
        '1': 1.0,    # Top pass rusher
        '2': 0.6,    # Second edge rusher
        '3': 0.3,    # Rotational
    },
    'LT': {
        '1': 1.0,    # Starting LT critical
        '2': 0.8,    # Backup LT still very important
    },
    'C': {
        '1': 1.0,    # Center calls protections
        '2': 0.7,    # Backup center significant
    }
}

# Injury Status Multipliers
# How likely a player is to miss the game or be limited
INJURY_STATUS_MULTIPLIERS: Dict[str, float] = {
    'Out': 1.0,                    # 100% out
    'Doubtful': 0.75,              # ~75% likely to miss
    'Questionable': 0.35,          # ~35% impact (50/50 to play, limited if plays)
    'Probable': 0.10,              # Minimal impact (was removed from injury reports)
    'IR': 1.0,                     # Injured Reserve - out for season
    'IR-R': 1.0,                   # IR - can return
    'PUP': 1.0,                    # Physically Unable to Perform
    'Suspended': 1.0,              # Suspension - same as Out
    'COV': 1.0,                    # COVID list (historical)
    'DNP': 0.50,                   # Did Not Practice
    'Limited': 0.20,               # Limited Practice
    'Full': 0.0,                   # Full Practice - no impact
}

# Premium Position Combinations
# Losing multiple key players compounds impact
PREMIUM_POSITIONS = ['QB', 'LT', 'EDGE', 'CB']

def get_position_weight(position: str, depth_rank: str = '1') -> float:
    """
    Get weighted impact for a position and depth rank.
    
    Args:
        position: Player position (QB, WR, LT, etc.)
        depth_rank: Depth chart rank ('1' for starter, '2' for backup, etc.)
    
    Returns:
        Float representing impact weight (0.0 - 1.0)
    """
    # Get base position weight
    base_weight = POSITION_WEIGHTS.get(position, 0.03)  # Default to minimal impact
    
    # Apply depth-specific adjustments if available
    if position in DEPTH_POSITION_ADJUSTMENTS:
        depth_mult = DEPTH_POSITION_ADJUSTMENTS[position].get(depth_rank, 0.1)
    else:
        depth_mult = DEPTH_MULTIPLIERS.get(depth_rank, 0.25)
    
    return base_weight * depth_mult


def get_injury_impact(
    position: str, 
    depth_rank: str, 
    injury_status: str,
    snap_percentage: float = None
) -> float:
    """
    Calculate total injury impact for a player.
    
    Args:
        position: Player position
        depth_rank: Depth chart rank
        injury_status: Injury status (Out, Doubtful, Questionable, etc.)
        snap_percentage: Optional snap % to further adjust impact
    
    Returns:
        Float representing total injury impact (0.0 - 1.0)
    """
    # Get position + depth weight
    position_impact = get_position_weight(position, depth_rank)
    
    # Get injury status multiplier
    status_mult = INJURY_STATUS_MULTIPLIERS.get(injury_status, 0.0)
    
    # Calculate base impact
    total_impact = position_impact * status_mult
    
    # Adjust by snap percentage if available
    if snap_percentage is not None:
        # If player normally plays 90% of snaps, they're more important than depth chart suggests
        snap_adjustment = snap_percentage / 100.0
        total_impact *= (0.5 + 0.5 * snap_adjustment)  # Weight snap % at 50%
    
    return min(total_impact, 1.0)  # Cap at 1.0


def categorize_injury_severity(impact_score: float) -> str:
    """
    Categorize overall injury impact into severity levels.
    
    Args:
        impact_score: Combined injury impact score for team
    
    Returns:
        String severity level
    """
    if impact_score >= 0.40:
        return "CRITICAL"      # Multiple key starters or QB out
    elif impact_score >= 0.25:
        return "SEVERE"        # Key starter(s) out
    elif impact_score >= 0.15:
        return "MODERATE"      # Important player(s) out
    elif impact_score >= 0.08:
        return "MINOR"         # Depth/rotational players
    else:
        return "MINIMAL"       # Questionable players or non-impact positions


# Key position flags for feature engineering
KEY_POSITION_FLAGS = {
    'qb_out': ['QB'],
    'top_wr_out': ['WR'],  # Will filter to WR1 only
    'top_cb_out': ['CB'],  # Will filter to CB1 only
    'lt_out': ['LT'],
    'edge_out': ['EDGE', 'DE'],
    'multiple_ol_out': ['LT', 'LG', 'C', 'RG', 'RT', 'OL'],  # Count >= 2
}


if __name__ == "__main__":
    # Test injury impact calculations
    print("🏈 Injury Impact Weight Testing\n")
    
    # Test 1: Starting QB out
    qb1_impact = get_injury_impact('QB', '1', 'Out')
    print(f"Starting QB Out: {qb1_impact:.3f} (35% impact expected)")
    
    # Test 2: Backup RB questionable
    rb2_impact = get_injury_impact('RB', '2', 'Questionable')
    print(f"Backup RB Questionable: {rb2_impact:.3f} (minimal impact expected)")
    
    # Test 3: Elite WR1 with high snap count
    wr1_impact = get_injury_impact('WR', '1', 'Out', snap_percentage=95.0)
    print(f"WR1 Out (95% snaps): {wr1_impact:.3f}")
    
    # Test 4: Multiple scenarios
    print("\n📊 Injury Severity Examples:")
    scenarios = [
        (0.35, "Starting QB Out"),
        (0.48, "QB + LT + WR1 Out"),
        (0.18, "Top EDGE + CB1 Questionable"),
        (0.06, "Backup RB + TE2 Out"),
    ]
    
    for score, desc in scenarios:
        severity = categorize_injury_severity(score)
        print(f"  {severity:12} - {desc} ({score:.2f})")








