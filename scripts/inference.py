"""
Inference Script
Interactive chat with automotive expert assistant.
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
from inference.chat import start_chat


def main():
    """Main inference pipeline."""
    
    print_banner("🚗 AUTOMOTIVE EXPERT ASSISTANT - CHAT INTERFACE")
    
    # Set seed
    set_seed(config.SEED)
    
    # Print GPU info
    if config.VERBOSE:
        print_gpu_info()
    
    # ========================================================================
    # Load Fine-Tuned Model
    # ========================================================================
    print("Loading automotive expert model...")
    
    model_path = config.MODEL_SAVE_DIR
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found at: {model_path}")
        print(f"\nPlease train the model first:")
        print(f"  python scripts/train.py")
        print(f"\nOr if you've saved it elsewhere, update MODEL_SAVE_DIR in config.py")
        sys.exit(1)
    
    print(f"Model location: {model_path}")
    print("Loading... (this may take a minute)\n")
    
    try:
        model, tokenizer = ModelLoader.load_for_inference(model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("\nMake sure the model was trained and saved successfully.")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========================================================================
    # Start Interactive Chat
    # ========================================================================
    start_chat(model, tokenizer)
    
    print("\n" + "="*70)
    print("Thanks for using Automotive Expert Assistant!")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Chat session ended. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error:")
        print(f"{e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
