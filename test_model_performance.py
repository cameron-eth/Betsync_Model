#!/usr/bin/env python3
"""
Model Performance Testing & Analysis
Tests the model against backfilled actual results and generates insights
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from supabase import create_client, Client
from dotenv import load_dotenv
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error,
    r2_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

load_dotenv()

# Configuration
SUPABASE_URL = os.getenv('SUPABASE_URL') or os.getenv('NEXT_PUBLIC_SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY') or os.getenv('SUPABASE_ANON_KEY') or os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY')


class ModelPerformanceTester:
    """Test and analyze model performance"""
    
    def __init__(self):
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("Missing Supabase credentials")
        
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    def fetch_completed_predictions(self, weeks_back: int = 4):
        """Fetch all completed predictions with results"""
        print("=" * 80)
        print("🏈 NFL MODEL PERFORMANCE TEST")
        print("=" * 80)
        print(f"📅 Analyzing last {weeks_back} weeks of predictions\n")
        
        cutoff_date = (datetime.now() - timedelta(weeks=weeks_back)).isoformat()
        
        try:
            response = self.supabase.table('nfl_model_predictions')\
                .select('*')\
                .eq('game_completed', True)\
                .gte('game_date', cutoff_date)\
                .execute()
            
            df = pd.DataFrame(response.data)
            print(f"✅ Found {len(df)} completed games with results\n")
            return df
            
        except Exception as e:
            print(f"❌ Error fetching predictions: {e}")
            return pd.DataFrame()
    
    def analyze_overall_performance(self, df: pd.DataFrame):
        """Analyze overall model accuracy"""
        print("=" * 80)
        print("📊 OVERALL MODEL PERFORMANCE")
        print("=" * 80)
        
        total_games = len(df)
        ml_correct = df['prediction_correct'].sum()
        spread_correct = df['spread_correct'].sum()
        
        ml_accuracy = (ml_correct / total_games * 100) if total_games > 0 else 0
        spread_accuracy = (spread_correct / total_games * 100) if total_games > 0 else 0
        
        print(f"\n🎯 Classification Accuracy:")
        print(f"   Total Games: {total_games}")
        print(f"   ML (Moneyline) Accuracy: {ml_correct}/{total_games} = {ml_accuracy:.1f}%")
        print(f"   Spread Accuracy: {spread_correct}/{total_games} = {spread_accuracy:.1f}%")
        
        return {
            'total_games': total_games,
            'ml_accuracy': ml_accuracy,
            'spread_accuracy': spread_accuracy
        }
    
    def calculate_statistical_metrics(self, df: pd.DataFrame):
        """Calculate R², RMSE, MAE, Brier Score, Log Loss"""
        print("\n" + "=" * 80)
        print("📐 STATISTICAL METRICS (ML Probabilities)")
        print("=" * 80)
        
        # Convert probabilities to numeric
        df['home_ml_prob_num'] = pd.to_numeric(df['home_ml_prob'], errors='coerce')
        df['home_cover_prob_num'] = pd.to_numeric(df['home_cover_prob'], errors='coerce')
        
        # Create actual outcomes (1 if home won, 0 if away won)
        y_actual_ml = (df['actual_winner'] == 'home').astype(int)
        y_actual_spread = (df['actual_spread_covered'] == 'home').astype(int)
        
        # Predicted probabilities
        y_pred_ml_prob = df['home_ml_prob_num'].values
        y_pred_spread_prob = df['home_cover_prob_num'].values
        
        try:
            # ML Model Metrics
            print("\n🎲 MONEYLINE MODEL:")
            
            # RMSE (Root Mean Squared Error)
            rmse_ml = np.sqrt(mean_squared_error(y_actual_ml, y_pred_ml_prob))
            print(f"   RMSE: {rmse_ml:.4f}")
            
            # MAE (Mean Absolute Error)
            mae_ml = mean_absolute_error(y_actual_ml, y_pred_ml_prob)
            print(f"   MAE: {mae_ml:.4f}")
            
            # R² Score (how well probabilities predict outcomes)
            r2_ml = r2_score(y_actual_ml, y_pred_ml_prob)
            print(f"   R² Score: {r2_ml:.4f}")
            
            # Brier Score (lower is better, range 0-1)
            brier_ml = brier_score_loss(y_actual_ml, y_pred_ml_prob)
            print(f"   Brier Score: {brier_ml:.4f} (lower is better)")
            
            # Log Loss
            logloss_ml = log_loss(y_actual_ml, np.column_stack([1-y_pred_ml_prob, y_pred_ml_prob]))
            print(f"   Log Loss: {logloss_ml:.4f} (lower is better)")
            
            # ROC AUC
            roc_auc_ml = roc_auc_score(y_actual_ml, y_pred_ml_prob)
            print(f"   ROC AUC: {roc_auc_ml:.4f}")
            
            # Spread Model Metrics
            print("\n📏 SPREAD MODEL:")
            
            rmse_spread = np.sqrt(mean_squared_error(y_actual_spread, y_pred_spread_prob))
            print(f"   RMSE: {rmse_spread:.4f}")
            
            mae_spread = mean_absolute_error(y_actual_spread, y_pred_spread_prob)
            print(f"   MAE: {mae_spread:.4f}")
            
            r2_spread = r2_score(y_actual_spread, y_pred_spread_prob)
            print(f"   R² Score: {r2_spread:.4f}")
            
            brier_spread = brier_score_loss(y_actual_spread, y_pred_spread_prob)
            print(f"   Brier Score: {brier_spread:.4f} (lower is better)")
            
            logloss_spread = log_loss(y_actual_spread, np.column_stack([1-y_pred_spread_prob, y_pred_spread_prob]))
            print(f"   Log Loss: {logloss_spread:.4f} (lower is better)")
            
            roc_auc_spread = roc_auc_score(y_actual_spread, y_pred_spread_prob)
            print(f"   ROC AUC: {roc_auc_spread:.4f}")
            
            # Interpretation Guide
            print("\n📖 Metric Interpretation:")
            print("   RMSE/MAE: Average prediction error (lower = better)")
            print("   R²: How well probs explain outcomes (1.0 = perfect, 0 = baseline)")
            print("   Brier Score: Probability calibration (0 = perfect, 0.25 = coin flip)")
            print("   Log Loss: Penalizes confident wrong predictions (lower = better)")
            print("   ROC AUC: Discrimination ability (1.0 = perfect, 0.5 = random)")
            
        except Exception as e:
            print(f"\n⚠️  Could not calculate some metrics: {e}")
    
    def show_confusion_matrices(self, df: pd.DataFrame):
        """Show confusion matrices for ML and Spread"""
        print("\n" + "=" * 80)
        print("🔲 CONFUSION MATRICES")
        print("=" * 80)
        
        # ML Confusion Matrix
        y_actual_ml = (df['actual_winner'] == 'home').astype(int)
        y_pred_ml = (df['home_ml_prob'] > 0.5).astype(int)
        
        cm_ml = confusion_matrix(y_actual_ml, y_pred_ml)
        
        print("\n🎲 MONEYLINE PREDICTIONS:")
        print("                  Predicted")
        print("                Away    Home")
        print(f"Actual  Away  [{cm_ml[0,0]:3d}]   [{cm_ml[0,1]:3d}]")
        print(f"        Home  [{cm_ml[1,0]:3d}]   [{cm_ml[1,1]:3d}]")
        
        # Calculate precision, recall, F1
        tn, fp, fn, tp = cm_ml.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   F1 Score: {f1:.3f}")
        
        # Spread Confusion Matrix
        y_actual_spread = (df['actual_spread_covered'] == 'home').astype(int)
        y_pred_spread = (df['home_cover_prob'] > 0.5).astype(int)
        
        cm_spread = confusion_matrix(y_actual_spread, y_pred_spread)
        
        print("\n📏 SPREAD PREDICTIONS:")
        print("                  Predicted")
        print("                Away    Home")
        print(f"Actual  Away  [{cm_spread[0,0]:3d}]   [{cm_spread[0,1]:3d}]")
        print(f"        Home  [{cm_spread[1,0]:3d}]   [{cm_spread[1,1]:3d}]")
        
        tn, fp, fn, tp = cm_spread.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   F1 Score: {f1:.3f}")
    
    def analyze_by_confidence(self, df: pd.DataFrame):
        """Analyze performance by confidence level"""
        print("\n" + "=" * 80)
        print("🎖️  PERFORMANCE BY CONFIDENCE LEVEL")
        print("=" * 80)
        
        for confidence in ['HIGH', 'MEDIUM', 'LOW']:
            conf_df = df[df['confidence_level'] == confidence]
            if len(conf_df) == 0:
                continue
            
            ml_acc = (conf_df['prediction_correct'].sum() / len(conf_df) * 100) if len(conf_df) > 0 else 0
            spread_acc = (conf_df['spread_correct'].sum() / len(conf_df) * 100) if len(conf_df) > 0 else 0
            
            print(f"\n{confidence} CONFIDENCE ({len(conf_df)} games):")
            print(f"   ML Accuracy: {ml_acc:.1f}%")
            print(f"   Spread Accuracy: {spread_acc:.1f}%")
    
    def analyze_by_bet_quality(self, df: pd.DataFrame):
        """Analyze performance by bet quality"""
        print("\n" + "=" * 80)
        print("💎 PERFORMANCE BY BET QUALITY")
        print("=" * 80)
        
        for quality in ['EXCELLENT', 'VERY GOOD', 'GOOD', 'FAIR', 'POOR']:
            qual_df = df[df['bet_quality'] == quality]
            if len(qual_df) == 0:
                continue
            
            ml_acc = (qual_df['prediction_correct'].sum() / len(qual_df) * 100) if len(qual_df) > 0 else 0
            spread_acc = (qual_df['spread_correct'].sum() / len(qual_df) * 100) if len(qual_df) > 0 else 0
            
            print(f"\n{quality} ({len(qual_df)} games):")
            print(f"   ML Accuracy: {ml_acc:.1f}%")
            print(f"   Spread Accuracy: {spread_acc:.1f}%")
    
    def analyze_home_away_bias(self, df: pd.DataFrame):
        """Check for home/away prediction bias"""
        print("\n" + "=" * 80)
        print("🏠 HOME vs AWAY BIAS ANALYSIS")
        print("=" * 80)
        
        # Predictions favoring home
        home_favored = df[df['home_ml_prob'] > df['away_ml_prob']]
        home_correct = home_favored['prediction_correct'].sum()
        home_acc = (home_correct / len(home_favored) * 100) if len(home_favored) > 0 else 0
        
        # Predictions favoring away
        away_favored = df[df['away_ml_prob'] > df['home_ml_prob']]
        away_correct = away_favored['prediction_correct'].sum()
        away_acc = (away_correct / len(away_favored) * 100) if len(away_favored) > 0 else 0
        
        print(f"\nPredictions favoring HOME: {len(home_favored)} games ({home_acc:.1f}% correct)")
        print(f"Predictions favoring AWAY: {len(away_favored)} games ({away_acc:.1f}% correct)")
        
        if abs(home_acc - away_acc) > 10:
            print(f"\n⚠️  WARNING: {abs(home_acc - away_acc):.1f}% bias detected!")
    
    def analyze_edge_performance(self, df: pd.DataFrame):
        """Analyze if higher edge predictions are more accurate"""
        print("\n" + "=" * 80)
        print("📈 EDGE vs ACCURACY ANALYSIS")
        print("=" * 80)
        
        # Convert edge_percentage to numeric
        df['edge_pct'] = pd.to_numeric(df['edge_percentage'], errors='coerce')
        
        # Bins for edge analysis
        bins = [
            ('High Edge (>20%)', df[df['edge_pct'] > 20]),
            ('Medium Edge (10-20%)', df[(df['edge_pct'] >= 10) & (df['edge_pct'] <= 20)]),
            ('Low Edge (<10%)', df[df['edge_pct'] < 10])
        ]
        
        for label, edge_df in bins:
            if len(edge_df) == 0:
                continue
            
            spread_acc = (edge_df['spread_correct'].sum() / len(edge_df) * 100) if len(edge_df) > 0 else 0
            print(f"\n{label} ({len(edge_df)} games):")
            print(f"   Spread Accuracy: {spread_acc:.1f}%")
    
    def identify_model_weaknesses(self, df: pd.DataFrame):
        """Identify where the model struggles"""
        print("\n" + "=" * 80)
        print("🔍 MODEL WEAKNESSES & INSIGHTS")
        print("=" * 80)
        
        # Close games (spread < 3)
        close_games = df[abs(df['spread']) < 3]
        close_acc = (close_games['spread_correct'].sum() / len(close_games) * 100) if len(close_games) > 0 else 0
        print(f"\n📍 Close games (spread < 3): {len(close_games)} games, {close_acc:.1f}% accuracy")
        
        # Blowouts (spread > 7)
        blowouts = df[abs(df['spread']) > 7]
        blowout_acc = (blowouts['spread_correct'].sum() / len(blowouts) * 100) if len(blowouts) > 0 else 0
        print(f"📍 Blowouts (spread > 7): {len(blowouts)} games, {blowout_acc:.1f}% accuracy")
        
        # Upsets (underdog wins)
        upsets = df[
            ((df['home_ml_prob'] < 0.5) & (df['actual_winner'] == 'home')) |
            ((df['away_ml_prob'] < 0.5) & (df['actual_winner'] == 'away'))
        ]
        print(f"📍 Upsets (underdog wins): {len(upsets)} games")
    
    def generate_recommendations(self, df: pd.DataFrame):
        """Generate betting recommendations based on analysis"""
        print("\n" + "=" * 80)
        print("💡 BETTING RECOMMENDATIONS")
        print("=" * 80)
        
        # Best bet types
        excellent_bets = df[df['bet_quality'] == 'EXCELLENT']
        excellent_acc = (excellent_bets['spread_correct'].sum() / len(excellent_bets) * 100) if len(excellent_bets) > 0 else 0
        
        print(f"\n✅ EXCELLENT quality bets:")
        print(f"   {len(excellent_bets)} bets, {excellent_acc:.1f}% win rate")
        
        if excellent_acc > 60:
            print(f"   👍 RECOMMENDATION: Focus on EXCELLENT quality bets")
        else:
            print(f"   ⚠️  CAUTION: Even EXCELLENT bets only hit {excellent_acc:.1f}%")
        
        # Edge analysis
        high_edge = df[df['edge_pct'] > 20]
        high_edge_acc = (high_edge['spread_correct'].sum() / len(high_edge) * 100) if len(high_edge) > 0 else 0
        
        print(f"\n✅ High edge bets (>20%):")
        print(f"   {len(high_edge)} bets, {high_edge_acc:.1f}% win rate")
        
        if high_edge_acc > 65:
            print(f"   👍 RECOMMENDATION: High edge bets are profitable")
        else:
            print(f"   ⚠️  CAUTION: High edge doesn't guarantee success")
    
    def run_full_analysis(self, weeks_back: int = 4):
        """Run complete model performance analysis"""
        df = self.fetch_completed_predictions(weeks_back)
        
        if df.empty:
            print("❌ No completed predictions found")
            return
        
        # Run all analyses
        self.analyze_overall_performance(df)
        self.calculate_statistical_metrics(df)
        self.show_confusion_matrices(df)
        self.analyze_by_confidence(df)
        self.analyze_by_bet_quality(df)
        self.analyze_home_away_bias(df)
        self.analyze_edge_performance(df)
        self.identify_model_weaknesses(df)
        self.generate_recommendations(df)
        
        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 80 + "\n")


def main():
    """Main execution"""
    try:
        tester = ModelPerformanceTester()
        
        # Default: analyze last 4 weeks
        weeks = int(sys.argv[1]) if len(sys.argv) > 1 else 4
        
        tester.run_full_analysis(weeks_back=weeks)
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

