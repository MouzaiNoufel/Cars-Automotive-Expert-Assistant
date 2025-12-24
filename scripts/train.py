"""
Main Training Script
Executes the complete training pipeline for automotive expert assistant.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import config
from utils.helpers import (
    set_seed,
    print_gpu_info,
    check_system_requirements,
    create_output_directories,
    print_banner
)
from data.dataset_generator import AutomotiveDatasetGenerator
from data.preprocessor import create_training_datasets
from models.model_loader import load_model_and_tokenizer
from training.trainer import create_trainer


def main():
    """Main training pipeline."""
    
    # Print configuration
    print_banner("🚗 CARS & AUTOMOTIVE EXPERT ASSISTANT - TRAINING")
    config.print_config_summary()
    
    # System checks
    check_system_requirements()
    print_gpu_info()
    
    # Set random seed for reproducibility
    set_seed(config.SEED)
    
    # Create output directories
    create_output_directories()
    
    # ========================================================================
    # STEP 1: Generate Dataset
    # ========================================================================
    print_banner("STEP 1: GENERATING AUTOMOTIVE DATASET")
    
    print(f"Generating {config.NUM_TRAINING_EXAMPLES} automotive instruction examples...")
    generator = AutomotiveDatasetGenerator(seed=config.SEED)
    raw_data = generator.generate_dataset(num_examples=config.NUM_TRAINING_EXAMPLES)
    
    print(f"✓ Generated {len(raw_data)} examples")
    print(f"\nSample example:")
    print(f"  Instruction: {raw_data[0]['instruction']}")
    print(f"  Output: {raw_data[0]['output'][:100]}...")
    
    # ========================================================================
    # STEP 2: Load Model and Tokenizer
    # ========================================================================
    print_banner("STEP 2: LOADING MODEL WITH QUANTIZATION AND LORA")
    
    print("Loading model and tokenizer...")
    print("This may take a few minutes on first run (downloading model)...")
    
    model, tokenizer = load_model_and_tokenizer()
    
    # ========================================================================
    # STEP 3: Prepare Dataset
    # ========================================================================
    print_banner("STEP 3: PREPROCESSING AND TOKENIZING DATASET")
    
    train_dataset, val_dataset, preprocessor = create_training_datasets(
        raw_data,
        tokenizer
    )
    
    print(f"✓ Dataset prepared:")
    print(f"  Training examples: {len(train_dataset)}")
    print(f"  Validation examples: {len(val_dataset)}")
    
    # ========================================================================
    # STEP 4: Setup Trainer
    # ========================================================================
    print_banner("STEP 4: SETTING UP TRAINER")
    
    trainer = create_trainer(model, tokenizer, train_dataset, val_dataset)
    trainer.print_training_summary()
    
    # ========================================================================
    # STEP 5: Train Model
    # ========================================================================
    print_banner("STEP 5: TRAINING MODEL")
    
    print("Starting training...")
    print("You can monitor progress in real-time.")
    print(f"TensorBoard logs will be saved to: {config.LOGS_DIR}")
    print(f"To view: tensorboard --logdir {config.LOGS_DIR}\n")
    
    # Train
    train_result = trainer.train()
    
    if train_result is not None:
        # ========================================================================
        # STEP 6: Evaluate Model
        # ========================================================================
        print_banner("STEP 6: EVALUATING MODEL")
        
        eval_metrics = trainer.evaluate()
        
        # ========================================================================
        # STEP 7: Save Model
        # ========================================================================
        print_banner("STEP 7: SAVING MODEL")
        
        trainer.save_model(config.MODEL_SAVE_DIR)
        
        # ========================================================================
        # COMPLETION
        # ========================================================================
        print_banner("✓ TRAINING COMPLETE!")
        
        print("🎉 Congratulations! Your automotive expert assistant is ready!")
        print(f"\n📁 Model saved to: {config.MODEL_SAVE_DIR}")
        print(f"📁 Training results: {config.OUTPUT_DIR}")
        print(f"📁 Logs: {config.LOGS_DIR}")
        
        print(f"\n📊 Final Metrics:")
        print(f"  Training Loss: {train_result.metrics.get('train_loss', 'N/A')}")
        print(f"  Validation Loss: {eval_metrics.get('eval_loss', 'N/A')}")
        
        print(f"\n🚀 Next Steps:")
        print(f"  1. Evaluate: python scripts/evaluate.py")
        print(f"  2. Chat: python scripts/inference.py")
        print(f"  3. Review logs: tensorboard --logdir {config.LOGS_DIR}")
        
        print("\n" + "="*70)
        print("Thank you for using the Automotive Expert Assistant trainer!")
        print("="*70 + "\n")
        
    else:
        print("\n⚠️  Training was interrupted or failed.")
        print("Check the error messages above for details.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user (Ctrl+C)")
        print("Partial progress may be saved in checkpoints.")
    except Exception as e:
        print(f"\n❌ Training failed with error:")
        print(f"{e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
