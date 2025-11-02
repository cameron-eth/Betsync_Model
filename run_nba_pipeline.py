#!/usr/bin/env python3
"""
NBA Hybrid Model Pipeline - Main Entry Point
Streamlined pipeline for the NBA hybrid model
"""

import sys
import argparse
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.nba_hybrid_pipeline import NBAHybridPipeline
from src.nba_advanced_features import NBAAdvancedFeatureEngine
from src.nba_hybrid_model import NBAHybridModel
from src.nba_utils import print_section_header


def print_banner():
    """Print pipeline banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║        🏀 NBA Hybrid Model Pipeline 🏀                     ║
    ║                                                           ║
    ║    Advanced Features → Superior Predictions              ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_advanced_features():
    """Generate advanced features from nba_api data"""
    print_section_header("ADVANCED FEATURES GENERATION")
    print("🚀 Extracting NBA features from nba_api...")
    print("⏱️  This will take 5-10 minutes\n")
    
    start = time.time()
    
    try:
        engine = NBAAdvancedFeatureEngine()
        engine.load_all_data()
        engine.create_enhanced_dataset()
        
        elapsed = time.time() - start
        print(f"\n✅ Advanced features generated in {elapsed/60:.1f} minutes")
        return True
        
    except Exception as e:
        print(f"\n❌ Advanced features generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_hybrid_training():
    """Train the hybrid model"""
    print_section_header("NBA HYBRID MODEL TRAINING")
    print("🤖 Training hybrid model with advanced features...")
    print("⏱️  This will take 3-5 minutes\n")
    
    start = time.time()
    
    try:
        trainer = NBAHybridModel()
        trainer.run_full_training()
        
        elapsed = time.time() - start
        print(f"\n✅ Hybrid model training completed in {elapsed/60:.1f} minutes")
        return True
        
    except Exception as e:
        print(f"\n❌ Hybrid model training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_hybrid_predictions():
    """Run hybrid model predictions"""
    print_section_header("NBA HYBRID MODEL PREDICTIONS")
    print("🏀 Running predictions with hybrid model...")
    print("⏱️  This will take 1-2 minutes\n")
    
    start = time.time()
    
    try:
        pipeline = NBAHybridPipeline()
        pipeline.run_full_pipeline()
        
        elapsed = time.time() - start
        print(f"\n✅ Predictions completed in {elapsed:.1f} seconds")
        return True
        
    except Exception as e:
        print(f"\n❌ Predictions failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_data_exists():
    """Check if required data exists"""
    from src.nba_config import NBA_HISTORIC_MATCHUPS_CSV, NBA_ENHANCED_FEATURES_CLEAN_CSV
    
    if not NBA_HISTORIC_MATCHUPS_CSV.exists():
        print("⚠️  Historic matchups CSV not found!")
        print(f"   Expected: {NBA_HISTORIC_MATCHUPS_CSV}")
        print("   Will attempt to create from nba_api schedules")
        return True  # Not fatal, can create from API
    
    if not NBA_ENHANCED_FEATURES_CLEAN_CSV.exists():
        print("⚠️  Enhanced features not found!")
        print(f"   Expected: {NBA_ENHANCED_FEATURES_CLEAN_CSV}")
        print("   Run: python run_nba_pipeline.py --features")
        return False
    
    print(f"✅ Found enhanced features: {NBA_ENHANCED_FEATURES_CLEAN_CSV}")
    return True


def check_hybrid_model_exists():
    """Check if hybrid model exists"""
    from src.nba_config import NBA_ML_MODEL_PATH, NBA_SPREAD_MODEL_PATH
    
    return NBA_ML_MODEL_PATH.exists() and NBA_SPREAD_MODEL_PATH.exists()


def run_full_pipeline():
    """Run complete NBA hybrid pipeline"""
    print_banner()
    
    # Pre-flight checks
    print_section_header("PRE-FLIGHT CHECKS")
    
    if not check_data_exists():
        return False
    
    print("\n📋 Pipeline Steps:")
    print("  1. Advanced Features (5-10 min)")
    print("  2. Hybrid Model Training (3-5 min)")
    print("  3. Predictions (1-2 min)")
    print("\n⏱️  Total estimated time: 10-17 minutes\n")
    
    input("Press ENTER to start pipeline...")
    print()
    
    # Step 1: Advanced Features
    if not run_advanced_features():
        print("\n❌ Pipeline failed at advanced features")
        return False
    
    print("\n" + "="*60 + "\n")
    
    # Step 2: Hybrid Model Training
    if not run_hybrid_training():
        print("\n❌ Pipeline failed at hybrid training")
        return False
    
    print("\n" + "="*60 + "\n")
    
    # Step 3: Predictions
    if not run_hybrid_predictions():
        print("\n❌ Pipeline failed at predictions")
        return False
    
    # Success
    print_section_header("NBA HYBRID PIPELINE COMPLETE! 🎉")
    print("✅ Advanced features: DONE")
    print("✅ Hybrid model training: DONE")
    print("✅ Predictions: DONE")
    
    print("\n📁 Output Files:")
    from src.nba_config import (
        NBA_ENHANCED_FEATURES_CLEAN_CSV, NBA_ML_MODEL_PATH,
        NBA_SPREAD_MODEL_PATH, NBA_FEATURE_COLUMNS_PATH, NBA_METADATA_PATH
    )
    print(f"  - {NBA_ENHANCED_FEATURES_CLEAN_CSV}")
    print(f"  - {NBA_ML_MODEL_PATH}")
    print(f"  - {NBA_SPREAD_MODEL_PATH}")
    print(f"  - {NBA_FEATURE_COLUMNS_PATH}")
    print(f"  - {NBA_METADATA_PATH}")
    
    print("\n🚀 The NBA Hybrid Model is ready!")
    
    return True


def run_quick_check():
    """Quick status check"""
    print_section_header("NBA HYBRID PIPELINE STATUS")
    
    print("📊 Checking data and models...\n")
    
    data_exists = check_data_exists()
    model_exists = check_hybrid_model_exists()
    
    status = []
    from src.nba_config import NBA_HISTORIC_MATCHUPS_CSV, NBA_ENHANCED_FEATURES_CLEAN_CSV
    status.append(("Historic Data (CSV)", NBA_HISTORIC_MATCHUPS_CSV.exists()))
    status.append(("Enhanced Features", NBA_ENHANCED_FEATURES_CLEAN_CSV.exists()))
    status.append(("Hybrid Models", model_exists))
    
    print("\nStatus:")
    for name, exists in status:
        icon = "✅" if exists else "❌"
        print(f"  {icon} {name}")
    
    print("\n📋 Recommendations:")
    if not NBA_ENHANCED_FEATURES_CLEAN_CSV.exists():
        print("  → Run: python run_nba_pipeline.py --features")
    elif not model_exists:
        print("  → Run: python run_nba_pipeline.py --train")
    else:
        print("  ✅ All set! NBA hybrid model is ready")
        print("  → Run predictions: python run_nba_pipeline.py --predict")


def main():
    parser = argparse.ArgumentParser(
        description='NBA Hybrid Model Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_nba_pipeline.py              # Run complete pipeline
  python run_nba_pipeline.py --features   # Generate advanced features only
  python run_nba_pipeline.py --train      # Train hybrid model only
  python run_nba_pipeline.py --predict    # Run predictions only
  python run_nba_pipeline.py --check      # Check status
        """
    )
    
    parser.add_argument(
        '--features',
        action='store_true',
        help='Generate advanced features from nba_api data'
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train the hybrid model'
    )
    
    parser.add_argument(
        '--predict',
        action='store_true',
        help='Run predictions with hybrid model'
    )
    
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check pipeline status'
    )
    
    args = parser.parse_args()
    
    # Determine what to run
    if args.check:
        run_quick_check()
        
    elif args.features:
        print_banner()
        success = run_advanced_features()
        sys.exit(0 if success else 1)
        
    elif args.train:
        print_banner()
        if not check_data_exists():
            sys.exit(1)
        success = run_hybrid_training()
        sys.exit(0 if success else 1)
        
    elif args.predict:
        print_banner()
        if not check_hybrid_model_exists():
            print("❌ ERROR: Hybrid model not found!")
            print("   Train model first: python run_nba_pipeline.py --train")
            sys.exit(1)
        success = run_hybrid_predictions()
        sys.exit(0 if success else 1)
        
    else:
        # Run full pipeline
        success = run_full_pipeline()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

