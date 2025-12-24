"""
Trainer Module
Handles the fine-tuning process with Hugging Face Trainer.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from typing import Dict
import config


class AutomotiveTrainer:
    """Manages the training process for automotive expert model."""
    
    def __init__(self, model, tokenizer, train_dataset, eval_dataset):
        """
        Initialize trainer.
        
        Args:
            model: Model with LoRA adapters
            tokenizer: Tokenizer for the model
            train_dataset: Tokenized training dataset
            eval_dataset: Tokenized evaluation dataset
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.trainer = None
        
    def _get_training_arguments(self) -> TrainingArguments:
        """
        Create training arguments from config.
        
        Returns:
            TrainingArguments object
        """
        args_dict = config.get_training_args()
        
        training_args = TrainingArguments(**args_dict)
        
        return training_args
    
    def _get_data_collator(self):
        """
        Create data collator for language modeling.
        
        Returns:
            DataCollatorForLanguageModeling
        """
        # Data collator handles batching and creates labels for causal LM
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
        )
        
        return data_collator
    
    def setup_trainer(self):
        """
        Set up Hugging Face Trainer.
        
        Returns:
            Configured Trainer
        """
        print("\n🎯 Setting up trainer...")
        
        # Get training arguments
        training_args = self._get_training_arguments()
        
        # Get data collator
        data_collator = self._get_data_collator()
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
        )
        
        print(f"✓ Trainer configured")
        print(f"  Output directory: {training_args.output_dir}")
        print(f"  Training examples: {len(self.train_dataset)}")
        print(f"  Validation examples: {len(self.eval_dataset)}")
        print(f"  Epochs: {training_args.num_train_epochs}")
        print(f"  Batch size per device: {training_args.per_device_train_batch_size}")
        print(f"  Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
        print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
        print(f"  Learning rate: {training_args.learning_rate}")
        print(f"  Warmup ratio: {training_args.warmup_ratio}")
        print(f"  Weight decay: {training_args.weight_decay}")
        print(f"  Max gradient norm: {training_args.max_grad_norm}")
        
        self.trainer = trainer
        return trainer
    
    def train(self):
        """
        Execute the training process.
        
        Returns:
            Training result
        """
        if self.trainer is None:
            self.setup_trainer()
        
        print("\n" + "="*70)
        print("🚀 STARTING TRAINING")
        print("="*70)
        print("\nTraining automotive expert assistant on car knowledge...")
        print("This may take a while depending on your hardware.\n")
        
        # Start training
        try:
            train_result = self.trainer.train()
            
            print("\n" + "="*70)
            print("✓ TRAINING COMPLETED SUCCESSFULLY!")
            print("="*70)
            
            # Print training metrics
            metrics = train_result.metrics
            print(f"\n📊 Final Training Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
            
            return train_result
            
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
            return None
        except Exception as e:
            print(f"\n❌ Training failed with error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate(self):
        """
        Evaluate the model on validation set.
        
        Returns:
            Evaluation metrics
        """
        if self.trainer is None:
            raise ValueError("Trainer not set up. Call setup_trainer() first.")
        
        print("\n" + "="*70)
        print("📊 EVALUATING MODEL")
        print("="*70)
        
        eval_result = self.trainer.evaluate()
        
        print(f"\n📊 Evaluation Metrics:")
        for key, value in eval_result.items():
            print(f"  {key}: {value}")
        
        return eval_result
    
    def save_model(self, output_dir: str = None):
        """
        Save the trained model.
        
        Args:
            output_dir: Directory to save model (default: from config)
        """
        if output_dir is None:
            output_dir = config.MODEL_SAVE_DIR
        
        if self.trainer is None:
            raise ValueError("Trainer not set up. Cannot save model.")
        
        print(f"\n💾 Saving model to {output_dir}...")
        
        # Save model (LoRA adapters)
        self.trainer.model.save_pretrained(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"✓ Model and tokenizer saved to {output_dir}")
        
        # Save training config
        import json
        config_dict = {
            'base_model': config.MODEL_NAME,
            'lora_r': config.LORA_R,
            'lora_alpha': config.LORA_ALPHA,
            'lora_dropout': config.LORA_DROPOUT,
            'target_modules': config.LORA_TARGET_MODULES,
            'learning_rate': config.LEARNING_RATE,
            'num_epochs': config.NUM_TRAIN_EPOCHS,
            'batch_size': config.PER_DEVICE_TRAIN_BATCH_SIZE,
            'max_seq_length': config.MAX_SEQ_LENGTH,
        }
        
        config_path = os.path.join(output_dir, 'training_config.json')
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"✓ Training configuration saved")
    
    def print_training_summary(self):
        """Print a summary of the training setup."""
        print("\n" + "="*70)
        print("📋 TRAINING SUMMARY")
        print("="*70)
        
        print(f"\n🎯 Objective:")
        print(f"  Fine-tune {config.MODEL_NAME} into automotive expert assistant")
        
        print(f"\n📊 Dataset:")
        print(f"  Training examples: {len(self.train_dataset)}")
        print(f"  Validation examples: {len(self.eval_dataset)}")
        
        print(f"\n🔧 Model Configuration:")
        print(f"  LoRA rank: {config.LORA_R}")
        print(f"  LoRA alpha: {config.LORA_ALPHA}")
        print(f"  Target modules: {', '.join(config.LORA_TARGET_MODULES)}")
        
        print(f"\n⚙️  Training Configuration:")
        print(f"  Epochs: {config.NUM_TRAIN_EPOCHS}")
        print(f"  Learning rate: {config.LEARNING_RATE}")
        print(f"  Batch size: {config.PER_DEVICE_TRAIN_BATCH_SIZE}")
        print(f"  Gradient accumulation: {config.GRADIENT_ACCUMULATION_STEPS}")
        print(f"  Max sequence length: {config.MAX_SEQ_LENGTH}")
        
        if torch.cuda.is_available():
            print(f"\n💻 Hardware:")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  Total GPU memory: {total_memory:.2f} GB")
        
        print("="*70 + "\n")


def create_trainer(model, tokenizer, train_dataset, eval_dataset):
    """
    Convenience function to create and setup trainer.
    
    Args:
        model: Model with LoRA adapters
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        
    Returns:
        Configured AutomotiveTrainer
    """
    trainer = AutomotiveTrainer(model, tokenizer, train_dataset, eval_dataset)
    trainer.setup_trainer()
    return trainer


if __name__ == "__main__":
    print("This module should be imported, not run directly.")
    print("Use scripts/train.py to train the model.")
