#!/usr/bin/env python3
"""
Pure Hybrid Model Pipeline - THE model
Streamlined pipeline for the pure hybrid model that combines historical + advanced features
"""

import sys
import argparse
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.pure_hybrid_pipeline import PureHybridPipeline
from src.advanced_features import AdvancedFeatureEngine
from src.true_hybrid_model import TrueHybridModel
from src.utils import print_section_header


def print_banner():
    """Print pipeline banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║        🏈 Pure Hybrid Model Pipeline - THE Model 🏈       ║
    ║                                                           ║
    ║    Historical + Advanced Features → Superior Predictions  ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_advanced_features():
    """Generate advanced features from PBP/NGS data"""
    print_section_header("ADVANCED FEATURES GENERATION")
    print("🚀 Extracting PBP and Next Gen Stats features...")
    print("⏱️  This will take 5-10 minutes\n")
    
    start = time.time()
    
    try:
        engine = AdvancedFeatureEngine()
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
    """Train the pure hybrid model"""
    print_section_header("PURE HYBRID MODEL TRAINING")
    print("🤖 Training hybrid model with historical + advanced features...")
    print("⏱️  This will take 3-5 minutes\n")
    
    start = time.time()
    
    try:
        trainer = TrueHybridModel()
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
    """Run pure hybrid model predictions"""
    print_section_header("PURE HYBRID MODEL PREDICTIONS")
    print("🏈 Running Week 7 predictions with THE model...")
    print("⏱️  This will take 1-2 minutes\n")
    
    start = time.time()
    
    try:
        pipeline = PureHybridPipeline()
        pipeline.run_full_pipeline()
        
        elapsed = time.time() - start
        print(f"\n✅ Hybrid predictions completed in {elapsed:.1f} seconds")
        return True
        
    except Exception as e:
        print(f"\n❌ Hybrid predictions failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_data_exists():
    """Check if required data exists"""
    data_path = Path(__file__).parent / 'data' / 'nfl_historic_matchups.csv'
    enhanced_path = Path(__file__).parent / 'data' / 'enhanced_features_clean.csv'
    
    if not data_path.exists():
        print("❌ ERROR: Historic matchups CSV not found!")
        print(f"   Expected: {data_path}")
        return False
    
    if not enhanced_path.exists():
        print("❌ ERROR: Enhanced features not found!")
        print(f"   Expected: {enhanced_path}")
        print("   Run: python run_pipeline.py --features")
        return False
    
    print(f"✅ Found historic data: {data_path}")
    print(f"✅ Found enhanced features: {enhanced_path}")
    return True


def check_hybrid_model_exists():
    """Check if hybrid model exists"""
    models_dir = Path(__file__).parent / 'models' / 'trained_models'
    ml_path = models_dir / 'true_hybrid_hybrid_ml_model.joblib'
    spread_path = models_dir / 'true_hybrid_hybrid_spread_model.joblib'
    
    return ml_path.exists() and spread_path.exists()


def run_full_pipeline():
    """Run complete pure hybrid pipeline"""
    print_banner()
    
    # Pre-flight checks
    print_section_header("PRE-FLIGHT CHECKS")
    
    if not check_data_exists():
        return False
    
    print("\n📋 Pipeline Steps:")
    print("  1. Advanced Features (5-10 min)")
    print("  2. Hybrid Model Training (3-5 min)")
    print("  3. Week 7 Predictions (1-2 min)")
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
    print_section_header("PURE HYBRID PIPELINE COMPLETE! 🎉")
    print("✅ Advanced features: DONE")
    print("✅ Hybrid model training: DONE")
    print("✅ Week 7 predictions: DONE")
    
    print("\n📁 Output Files:")
    print("  - data/enhanced_features_clean.csv")
    print("  - models/trained_models/true_hybrid_hybrid_ml_model.joblib")
    print("  - models/trained_models/true_hybrid_hybrid_spread_model.joblib")
    print("  - models/trained_models/true_hybrid_feature_columns.json")
    print("  - models/trained_models/true_hybrid_metadata.json")
    
    print("\n🚀 The Pure Hybrid Model is THE model!")
    print("   - Combines historical weather/stadium data")
    print("   - Uses advanced PBP/NGS features")
    print("   - 80%+ spread accuracy")
    print("   - Superior predictive power")
    
    return True


def run_quick_check():
    """Quick status check"""
    print_section_header("PURE HYBRID PIPELINE STATUS")
    
    print("📊 Checking data and models...\n")
    
    data_exists = check_data_exists()
    model_exists = check_hybrid_model_exists()
    
    status = []
    status.append(("Historic Data (CSV)", Path(__file__).parent / 'data' / 'nfl_historic_matchups.csv').exists())
    status.append(("Enhanced Features", Path(__file__).parent / 'data' / 'enhanced_features_clean.csv').exists())
    status.append(("Hybrid Models", model_exists))
    
    print("\nStatus:")
    for name, exists in status:
        icon = "✅" if exists else "❌"
        print(f"  {icon} {name}")
    
    print("\n📋 Recommendations:")
    if not data_exists:
        print("  ⚠️  Copy historic CSV to data/ directory first")
    elif not (Path(__file__).parent / 'data' / 'enhanced_features_clean.csv').exists():
        print("  → Run: python run_pipeline.py --features")
    elif not model_exists:
        print("  → Run: python run_pipeline.py --train")
    else:
        print("  ✅ All set! Pure hybrid model is ready")
        print("  → Run predictions: python run_pipeline.py --predict")


def main():
    parser = argparse.ArgumentParser(
        description='Pure Hybrid Model Pipeline - THE model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py              # Run complete pipeline
  python run_pipeline.py --features   # Generate advanced features only
  python run_pipeline.py --train      # Train hybrid model only
  python run_pipeline.py --predict    # Run Week 7 predictions only
  python run_pipeline.py --check      # Check status
        """
    )
    
    parser.add_argument(
        '--features',
        action='store_true',
        help='Generate advanced features from PBP/NGS data'
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train the pure hybrid model'
    )
    
    parser.add_argument(
        '--predict',
        action='store_true',
        help='Run Week 7 predictions with hybrid model'
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
        if not (Path(__file__).parent / 'data' / 'nfl_historic_matchups.csv').exists():
            print("❌ ERROR: Historic data not found!")
            sys.exit(1)
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
            print("   Train model first: python run_pipeline.py --train")
            sys.exit(1)
        success = run_hybrid_predictions()
        sys.exit(0 if success else 1)
        
    else:
        # Run full pipeline
        success = run_full_pipeline()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()