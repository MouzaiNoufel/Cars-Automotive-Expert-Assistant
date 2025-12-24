"""
Evaluation Script
Evaluates fine-tuned automotive expert model on test queries.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import config
from utils.helpers import set_seed, print_gpu_info, print_banner
from models.model_loader import ModelLoader
from evaluation.evaluator import evaluate_model


def main():
    """Main evaluation pipeline."""
    
    print_banner("🚗 AUTOMOTIVE EXPERT ASSISTANT - EVALUATION")
    
    # Set seed for reproducibility
    set_seed(config.SEED)
    
    # Print GPU info
    print_gpu_info()
    
    # ========================================================================
    # Load Fine-Tuned Model
    # ========================================================================
    print_banner("LOADING FINE-TUNED MODEL")
    
    model_path = config.MODEL_SAVE_DIR
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print(f"\nPlease train the model first:")
        print(f"  python scripts/train.py")
        sys.exit(1)
    
    print(f"Loading model from: {model_path}")
    
    try:
        model, tokenizer = ModelLoader.load_for_inference(model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("\nMake sure you have completed training successfully.")
        sys.exit(1)
    
    # ========================================================================
    # Run Evaluation
    # ========================================================================
    print_banner("RUNNING EVALUATION")
    
    print("Testing model on automotive domain questions...")
    print("This will evaluate the model on predefined automotive queries.\n")
    
    results = evaluate_model(model, tokenizer)
    
    # ========================================================================
    # Summary
    # ========================================================================
    print_banner("EVALUATION COMPLETE")
    
    print(f"✓ Evaluated on {len(results)} queries")
    print(f"\nResults show the model's ability to:")
    print(f"  • Explain automotive concepts")
    print(f"  • Provide buying recommendations")
    print(f"  • Compare vehicles")
    print(f"  • Answer maintenance questions")
    print(f"  • Discuss electric and hybrid technologies")
    
    print(f"\n🚀 Next Steps:")
    print(f"  • Try interactive chat: python scripts/inference.py")
    print(f"  • Test with your own questions in chat mode")
    print(f"  • Review training logs: tensorboard --logdir {config.LOGS_DIR}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Evaluation failed with error:")
        print(f"{e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
