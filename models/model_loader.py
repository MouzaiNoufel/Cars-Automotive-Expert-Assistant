"""
Model Loader with Quantization and LoRA Configuration
Loads base LLM with 4-bit quantization and prepares for fine-tuning.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
import config


class ModelLoader:
    """Handles model loading with quantization and LoRA configuration."""
    
    def __init__(self):
        """Initialize model loader with config settings."""
        self.model_name = config.MODEL_NAME
        self.device = config.DEVICE
        self.model = None
        self.tokenizer = None
        self.peft_config = None
    
    def load_tokenizer(self):
        """
        Load and configure tokenizer.
        
        Returns:
            Configured tokenizer
        """
        print(f"📥 Loading tokenizer from {self.model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Set padding side to right for training
        tokenizer.padding_side = 'right'
        
        print(f"✓ Tokenizer loaded successfully")
        print(f"  Vocabulary size: {len(tokenizer)}")
        print(f"  Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        print(f"  EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
        
        self.tokenizer = tokenizer
        return tokenizer
    
    def _get_quantization_config(self):
        """
        Create quantization configuration for 4-bit training.
        
        Returns:
            BitsAndBytesConfig for 4-bit quantization
        """
        if not config.USE_4BIT_QUANTIZATION:
            return None
        
        # Configure 4-bit quantization with bitsandbytes
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Enable 4-bit loading
            bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,  # NF4 quantization
            bnb_4bit_compute_dtype=getattr(torch, config.BNB_4BIT_COMPUTE_DTYPE),  # bfloat16
            bnb_4bit_use_double_quant=config.USE_NESTED_QUANT,  # Nested quantization
        )
        
        return bnb_config
    
    def load_base_model(self):
        """
        Load base model with quantization.
        
        Returns:
            Loaded and quantized base model
        """
        print(f"\n📥 Loading base model: {self.model_name}")
        print(f"   Device: {self.device}")
        
        # Get quantization config
        bnb_config = self._get_quantization_config()
        
        if bnb_config:
            print(f"   Quantization: 4-bit {config.BNB_4BIT_QUANT_TYPE.upper()}")
            print(f"   Compute dtype: {config.BNB_4BIT_COMPUTE_DTYPE}")
        else:
            print(f"   Quantization: None (full precision)")
        
        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",  # Automatically distribute across GPUs
            trust_remote_code=True,
            torch_dtype=config.TORCH_DTYPE,
        )
        
        # Disable cache for gradient checkpointing
        model.config.use_cache = False
        model.config.pretraining_tp = 1  # Tensor parallelism
        
        print(f"✓ Base model loaded successfully")
        
        # Print memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
        
        self.model = model
        return model
    
    def _create_lora_config(self):
        """
        Create LoRA configuration for parameter-efficient fine-tuning.
        
        Returns:
            LoraConfig object
        """
        lora_config = LoraConfig(
            r=config.LORA_R,  # Rank of LoRA matrices
            lora_alpha=config.LORA_ALPHA,  # Scaling factor
            target_modules=config.LORA_TARGET_MODULES,  # Which layers to adapt
            lora_dropout=config.LORA_DROPOUT,  # Dropout for regularization
            bias=config.LORA_BIAS,  # Bias training strategy
            task_type=config.LORA_TASK_TYPE,  # Causal language modeling
        )
        
        return lora_config
    
    def prepare_for_training(self):
        """
        Prepare model for training with LoRA.
        
        Returns:
            Model ready for training with LoRA adapters
        """
        if self.model is None:
            raise ValueError("Base model not loaded. Call load_base_model() first.")
        
        print(f"\n🔧 Preparing model for LoRA fine-tuning...")
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Create LoRA configuration
        self.peft_config = self._create_lora_config()
        
        print(f"   LoRA Configuration:")
        print(f"     Rank (r): {config.LORA_R}")
        print(f"     Alpha: {config.LORA_ALPHA}")
        print(f"     Dropout: {config.LORA_DROPOUT}")
        print(f"     Target modules: {', '.join(config.LORA_TARGET_MODULES)}")
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, self.peft_config)
        
        # Print trainable parameters
        self._print_trainable_parameters()
        
        return self.model
    
    def _print_trainable_parameters(self):
        """Print the number of trainable parameters."""
        trainable_params = 0
        all_params = 0
        
        for _, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        trainable_percent = 100 * trainable_params / all_params
        
        print(f"\n   📊 Trainable Parameters:")
        print(f"      Total params: {all_params:,}")
        print(f"      Trainable params: {trainable_params:,}")
        print(f"      Trainable %: {trainable_percent:.4f}%")
        print(f"      Memory savings: {100 - trainable_percent:.2f}%")
        
        print(f"\n✓ Model ready for training!")
    
    def load_for_training(self):
        """
        Complete loading pipeline: tokenizer + model + LoRA.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        # Load tokenizer
        tokenizer = self.load_tokenizer()
        
        # Load base model with quantization
        model = self.load_base_model()
        
        # Prepare for LoRA training
        model = self.prepare_for_training()
        
        return model, tokenizer
    
    def save_model(self, output_dir: str):
        """
        Save fine-tuned LoRA adapters and tokenizer.
        
        Args:
            output_dir: Directory to save model and tokenizer
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before saving.")
        
        print(f"\n💾 Saving model to {output_dir}...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save LoRA adapters
        self.model.save_pretrained(output_dir)
        print(f"   ✓ LoRA adapters saved")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        print(f"   ✓ Tokenizer saved")
        
        # Save config for reference
        import json
        config_dict = {
            'base_model': self.model_name,
            'lora_r': config.LORA_R,
            'lora_alpha': config.LORA_ALPHA,
            'lora_dropout': config.LORA_DROPOUT,
            'target_modules': config.LORA_TARGET_MODULES,
            'max_seq_length': config.MAX_SEQ_LENGTH,
        }
        
        config_path = os.path.join(output_dir, 'training_config.json')
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"   ✓ Training config saved")
        
        print(f"✓ Model saved successfully to {output_dir}")
    
    @staticmethod
    def load_for_inference(model_path: str, device: str = None):
        """
        Load fine-tuned model for inference.
        
        Args:
            model_path: Path to saved LoRA model
            device: Device to load on (default: from config)
            
        Returns:
            Tuple of (model, tokenizer)
        """
        from peft import PeftModel
        
        if device is None:
            device = config.DEVICE
        
        print(f"\n📥 Loading fine-tuned model from {model_path}...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"   ✓ Tokenizer loaded")
        
        # Load base model with quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config.USE_4BIT_QUANTIZATION,
            bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
            bnb_4bit_compute_dtype=getattr(torch, config.BNB_4BIT_COMPUTE_DTYPE),
            bnb_4bit_use_double_quant=config.USE_NESTED_QUANT,
        ) if config.USE_4BIT_QUANTIZATION else None
        
        base_model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"   ✓ Base model loaded")
        
        # Load LoRA adapters
        model = PeftModel.from_pretrained(base_model, model_path)
        print(f"   ✓ LoRA adapters loaded")
        
        # Set to evaluation mode
        model.eval()
        
        print(f"✓ Model ready for inference!")
        
        return model, tokenizer


def load_model_and_tokenizer():
    """
    Convenience function to load model and tokenizer for training.
    
    Returns:
        Tuple of (model, tokenizer)
    """
    loader = ModelLoader()
    return loader.load_for_training()


if __name__ == "__main__":
    # Test model loading
    print("="*70)
    print("TESTING MODEL LOADING WITH QUANTIZATION AND LORA")
    print("="*70)
    
    try:
        model, tokenizer = load_model_and_tokenizer()
        print("\n" + "="*70)
        print("✓ MODEL LOADING TEST SUCCESSFUL!")
        print("="*70)
        
        # Test tokenization
        test_text = "What is the difference between horsepower and torque?"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"\nTest tokenization: '{test_text}'")
        print(f"Token IDs shape: {tokens['input_ids'].shape}")
        print(f"First 10 tokens: {tokens['input_ids'][0][:10].tolist()}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
