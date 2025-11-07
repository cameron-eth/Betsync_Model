#!/usr/bin/env python3
"""
Profitability Analysis
Pull actual betting results from DB and calculate ROI/profit
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL') or os.getenv('NEXT_PUBLIC_SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY') or os.getenv('SUPABASE_ANON_KEY') or os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY')


class ProfitabilityAnalyzer:
    """Analyze actual betting profitability"""
    
    def __init__(self):
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("Missing Supabase credentials")
        
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    def american_to_decimal(self, american_odds):
        """Convert American odds to decimal (odds stored as basis points)"""
        # Odds are stored as basis points: 31800 = +318 American odds
        american_odds = american_odds / 100
        
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1
    
    def calculate_payout(self, stake, american_odds, won):
        """Calculate profit/loss for a bet"""
        if not won:
            return -stake
        
        decimal_odds = self.american_to_decimal(american_odds)
        return stake * (decimal_odds - 1)  # Profit only
    
    def fetch_completed_bets(self, weeks_back=4):
        """Fetch all completed predictions"""
        print("=" * 80)
        print("💰 NFL MODEL PROFITABILITY ANALYSIS")
        print("=" * 80)
        print(f"📅 Analyzing last {weeks_back} weeks\n")
        
        cutoff_date = (datetime.now() - timedelta(weeks=weeks_back)).isoformat()
        
        try:
            response = self.supabase.table('nfl_model_predictions')\
                .select('*')\
                .eq('game_completed', True)\
                .gte('game_date', cutoff_date)\
                .execute()
            
            df = pd.DataFrame(response.data)
            print(f"✅ Found {len(df)} completed predictions\n")
            return df
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return pd.DataFrame()
    
    def analyze_profitability(self, df, stake_per_bet=100):
        """Calculate profitability if we bet on model recommendations"""
        print("=" * 80)
        print(f"💵 PROFITABILITY ANALYSIS (${stake_per_bet} per bet)")
        print("=" * 80)
        
        # Filter for games we would have bet on (recommended_bet != "-")
        bets_df = df[df['recommended_bet'] != '-'].copy()
        
        print(f"\n📊 Betting Activity:")
        print(f"   Total games: {len(df)}")
        print(f"   Games we bet on: {len(bets_df)}")
        print(f"   Games we passed: {len(df) - len(bets_df)}")
        
        if len(bets_df) == 0:
            print("\n⚠️  No bets placed - model recommended passing on all games")
            return
        
        # Calculate profit for each bet
        total_profit = 0
        wins = 0
        losses = 0
        
        for _, row in bets_df.iterrows():
            # Determine which side we bet on
            if row['recommended_bet'] == 'HOME':
                won = row['spread_correct'] if row['actual_spread_covered'] == 'home' else False
                odds = row['home_ml_odds']
            elif row['recommended_bet'] == 'AWAY':
                won = row['spread_correct'] if row['actual_spread_covered'] == 'away' else False
                odds = row['away_ml_odds']
            else:
                continue
            
            # Calculate profit/loss
            profit = self.calculate_payout(stake_per_bet, odds, won)
            total_profit += profit
            
            if won:
                wins += 1
            else:
                losses += 1
        
        # Calculate metrics
        total_risked = len(bets_df) * stake_per_bet
        roi = (total_profit / total_risked * 100) if total_risked > 0 else 0
        win_rate = (wins / len(bets_df) * 100) if len(bets_df) > 0 else 0
        
        print(f"\n💰 RESULTS:")
        print(f"   Total Bets: {len(bets_df)}")
        print(f"   Wins: {wins}")
        print(f"   Losses: {losses}")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"\n   Total Risked: ${total_risked:,.2f}")
        print(f"   Total Profit/Loss: {'$' if total_profit >= 0 else '-$'}{abs(total_profit):,.2f}")
        print(f"   ROI: {roi:+.1f}%")
        
        if roi > 0:
            print(f"\n   ✅ PROFITABLE! ${total_profit:,.2f} profit")
        elif roi == 0:
            print(f"\n   ➖ Break even")
        else:
            print(f"\n   ❌ Loss of ${abs(total_profit):,.2f}")
        
        return {
            'total_profit': total_profit,
            'roi': roi,
            'win_rate': win_rate,
            'bets': len(bets_df)
        }
    
    def analyze_by_quality(self, df, stake_per_bet=100):
        """Break down profitability by bet quality"""
        print("\n" + "=" * 80)
        print("💎 PROFITABILITY BY BET QUALITY")
        print("=" * 80)
        
        for quality in ['EXCELLENT', 'VERY GOOD', 'GOOD', 'FAIR']:
            qual_df = df[(df['bet_quality'] == quality) & (df['recommended_bet'] != '-')]
            
            if len(qual_df) == 0:
                continue
            
            profit = 0
            wins = 0
            
            for _, row in qual_df.iterrows():
                if row['recommended_bet'] == 'HOME':
                    won = row['spread_correct'] if row['actual_spread_covered'] == 'home' else False
                    odds = row['home_ml_odds']
                else:
                    won = row['spread_correct'] if row['actual_spread_covered'] == 'away' else False
                    odds = row['away_ml_odds']
                
                profit += self.calculate_payout(stake_per_bet, odds, won)
                if won:
                    wins += 1
            
            roi = (profit / (len(qual_df) * stake_per_bet) * 100) if len(qual_df) > 0 else 0
            win_rate = (wins / len(qual_df) * 100) if len(qual_df) > 0 else 0
            
            print(f"\n{quality}:")
            print(f"   Bets: {len(qual_df)}")
            print(f"   Win Rate: {win_rate:.1f}%")
            print(f"   Profit: {'$' if profit >= 0 else '-$'}{abs(profit):,.2f}")
            print(f"   ROI: {roi:+.1f}%")
    
    def analyze_by_confidence(self, df, stake_per_bet=100):
        """Break down profitability by confidence level"""
        print("\n" + "=" * 80)
        print("🎖️  PROFITABILITY BY CONFIDENCE")
        print("=" * 80)
        
        for conf in ['HIGH', 'MEDIUM', 'LOW']:
            conf_df = df[(df['confidence_level'] == conf) & (df['recommended_bet'] != '-')]
            
            if len(conf_df) == 0:
                continue
            
            profit = 0
            wins = 0
            
            for _, row in conf_df.iterrows():
                if row['recommended_bet'] == 'HOME':
                    won = row['spread_correct'] if row['actual_spread_covered'] == 'home' else False
                    odds = row['home_ml_odds']
                else:
                    won = row['spread_correct'] if row['actual_spread_covered'] == 'away' else False
                    odds = row['away_ml_odds']
                
                profit += self.calculate_payout(stake_per_bet, odds, won)
                if won:
                    wins += 1
            
            roi = (profit / (len(conf_df) * stake_per_bet) * 100) if len(conf_df) > 0 else 0
            win_rate = (wins / len(conf_df) * 100) if len(conf_df) > 0 else 0
            
            print(f"\n{conf} CONFIDENCE:")
            print(f"   Bets: {len(conf_df)}")
            print(f"   Win Rate: {win_rate:.1f}%")
            print(f"   Profit: {'$' if profit >= 0 else '-$'}{abs(profit):,.2f}")
            print(f"   ROI: {roi:+.1f}%")
    
    def run_analysis(self, weeks_back=4, stake_per_bet=100):
        """Run complete profitability analysis"""
        df = self.fetch_completed_bets(weeks_back)
        
        if df.empty:
            print("❌ No data found")
            return
        
        self.analyze_profitability(df, stake_per_bet)
        self.analyze_by_quality(df, stake_per_bet)
        self.analyze_by_confidence(df, stake_per_bet)
        
        print("\n" + "=" * 80)
        print("✅ PROFITABILITY ANALYSIS COMPLETE")
        print("=" * 80)
        print("\n💡 Key Insight: Accuracy ≠ Profitability")
        print("   Statistical metrics (R², RMSE) measure calibration")
        print("   ROI measures what actually matters: making money!")
        print("=" * 80 + "\n")


def main():
    try:
        analyzer = ProfitabilityAnalyzer()
        
        weeks = int(sys.argv[1]) if len(sys.argv) > 1 else 4
        stake = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        
        analyzer.run_analysis(weeks_back=weeks, stake_per_bet=stake)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

