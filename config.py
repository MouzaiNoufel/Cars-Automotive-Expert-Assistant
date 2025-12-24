"""
Central Configuration for Cars & Automotive Expert Assistant Fine-Tuning
All hyperparameters and settings are defined here for easy experimentation.
"""

import os
import torch

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Base model selection (choose one)
# Option 1: Mistral-7B (recommended for quality/efficiency)
MODEL_NAME = "mistralai/Mistral-7B-v0.1"

# Option 2: Llama-2-7B (alternative, requires HuggingFace access)
# MODEL_NAME = "meta-llama/Llama-2-7b-hf"

# Quantization settings for memory efficiency
USE_4BIT_QUANTIZATION = True
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"  # Use bfloat16 for computation
BNB_4BIT_QUANT_TYPE = "nf4"  # NormalFloat 4-bit quantization
USE_NESTED_QUANT = True  # Double quantization for extra memory savings

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# ============================================================================
# LORA CONFIGURATION
# ============================================================================

# LoRA hyperparameters
LORA_R = 64  # Rank of LoRA matrices (higher = more capacity, more memory)
LORA_ALPHA = 16  # Scaling factor for LoRA (typically r/2 or r/4)
LORA_DROPOUT = 0.05  # Dropout for regularization

# Target modules to apply LoRA (attention layers)
# For Mistral/Llama: query, key, value, output projections
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]

# Task type
LORA_TASK_TYPE = "CAUSAL_LM"

# Bias training
LORA_BIAS = "none"  # Options: "none", "all", "lora_only"

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Training hyperparameters
LEARNING_RATE = 2e-4  # Optimized for LoRA fine-tuning
NUM_TRAIN_EPOCHS = 3  # Number of training epochs
PER_DEVICE_TRAIN_BATCH_SIZE = 4  # Batch size per GPU
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 4 * 4 = 16
WARMUP_RATIO = 0.03  # 3% of training for warmup
MAX_GRAD_NORM = 0.3  # Gradient clipping threshold
WEIGHT_DECAY = 0.001  # L2 regularization

# Optimization settings
OPTIM = "paged_adamw_8bit"  # 8-bit AdamW for memory efficiency
LR_SCHEDULER_TYPE = "cosine"  # Learning rate schedule
MAX_SEQ_LENGTH = 512  # Maximum sequence length for training

# Evaluation and logging
EVAL_STRATEGY = "steps"  # Evaluate every N steps
EVAL_STEPS = 50  # Evaluation frequency
LOGGING_STEPS = 10  # Log every N steps
SAVE_STEPS = 50  # Save checkpoint every N steps
SAVE_TOTAL_LIMIT = 3  # Keep only last 3 checkpoints

# Use new parameter name for newer transformers versions
EVALUATION_STRATEGY = EVAL_STRATEGY  # Backward compatibility

# Advanced training settings
FP16 = False  # Don't use FP16 (we use bfloat16 with quantization)
BF16 = torch.cuda.is_available()  # Use bfloat16 only on GPU, fp32 on CPU
GRADIENT_CHECKPOINTING = True  # Trade compute for memory
GROUP_BY_LENGTH = True  # Group sequences of similar length (faster training)

# Reproducibility
SEED = 42

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

# Dataset generation
NUM_TRAINING_EXAMPLES = 500  # Number of instruction-response pairs
TRAIN_TEST_SPLIT = 0.9  # 90% train, 10% validation
SHUFFLE_DATASET = True

# Prompt template for instruction formatting
PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

PROMPT_TEMPLATE_NO_INPUT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Output directories
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "automotive_expert_model")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Dataset cache
DATASET_CACHE_DIR = os.path.join(BASE_DIR, ".cache", "datasets")
os.makedirs(DATASET_CACHE_DIR, exist_ok=True)

# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================

# Generation parameters for inference
GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "do_sample": True,
    "pad_token_id": None,  # Will be set from tokenizer
    "eos_token_id": None,  # Will be set from tokenizer
}

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

# Test queries for automotive domain evaluation
EVALUATION_QUERIES = [
    {
        "instruction": "What's the difference between a turbocharged and supercharged engine?",
        "input": "",
    },
    {
        "instruction": "Compare the Toyota Camry and Honda Accord for a family.",
        "input": "I need a reliable sedan with good fuel economy and low maintenance costs.",
    },
    {
        "instruction": "Should I buy a hybrid or electric car?",
        "input": "I drive about 50 miles daily and have access to home charging.",
    },
    {
        "instruction": "Explain how regenerative braking works in electric vehicles.",
        "input": "",
    },
    {
        "instruction": "What are the most important safety features to look for in a new car?",
        "input": "",
    },
    {
        "instruction": "Best used car under $15,000?",
        "input": "Looking for something reliable with low running costs.",
    },
    {
        "instruction": "How often should I change my car's oil?",
        "input": "I drive a 2020 Honda Civic with synthetic oil.",
    },
    {
        "instruction": "What's the difference between AWD and 4WD?",
        "input": "",
    },
]

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Logging level
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR

# Report to (for experiment tracking)
REPORT_TO = "tensorboard"  # Options: "tensorboard", "wandb", "none"

# ============================================================================
# HARDWARE & PERFORMANCE
# ============================================================================

# DataLoader settings
DATALOADER_NUM_WORKERS = 4  # Number of CPU workers for data loading
DATALOADER_PIN_MEMORY = True  # Pin memory for faster GPU transfer

# Mixed precision training
AUTO_FIND_BATCH_SIZE = False  # Automatically find max batch size (experimental)

# ============================================================================
# DISPLAY & UI
# ============================================================================

# Progress bars
DISABLE_TQDM = False  # Show progress bars during training

# Console output
VERBOSE = True  # Print detailed information

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration settings and print warnings if needed."""
    
    if not torch.cuda.is_available() and USE_4BIT_QUANTIZATION:
        print("⚠️  WARNING: CUDA not available. Quantization may not work on CPU.")
        print("   Training will be slow. Consider using a GPU.")
    
    if LORA_R < LORA_ALPHA:
        print(f"⚠️  WARNING: LoRA rank ({LORA_R}) is less than alpha ({LORA_ALPHA}).")
        print("   This is unusual. Consider setting alpha = rank or rank/2.")
    
    if PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS < 8:
        print(f"⚠️  WARNING: Effective batch size is {PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}.")
        print("   This is quite small. Training may be unstable.")
    
    if MAX_SEQ_LENGTH > 2048:
        print(f"⚠️  WARNING: Max sequence length ({MAX_SEQ_LENGTH}) is very long.")
        print("   This will significantly increase memory usage.")
    
    print("✓ Configuration validated successfully!")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_training_args():
    """Return a dictionary of training arguments for Hugging Face Trainer."""
    return {
        "output_dir": OUTPUT_DIR,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "warmup_ratio": WARMUP_RATIO,
        "logging_steps": LOGGING_STEPS,
        "save_steps": SAVE_STEPS,
        "eval_steps": EVAL_STEPS,
        "eval_strategy": EVAL_STRATEGY,  # Updated from evaluation_strategy
        "save_total_limit": SAVE_TOTAL_LIMIT,
        "fp16": FP16,
        "bf16": BF16,
        "gradient_checkpointing": GRADIENT_CHECKPOINTING,
        "group_by_length": GROUP_BY_LENGTH,
        "max_grad_norm": MAX_GRAD_NORM,
        "weight_decay": WEIGHT_DECAY,
        "optim": OPTIM,
        "lr_scheduler_type": LR_SCHEDULER_TYPE,
        "seed": SEED,
        "report_to": REPORT_TO,
        "logging_dir": LOGS_DIR,
        "load_best_model_at_end": True,
        "metric_for_best_model": "loss",
        "greater_is_better": False,
        "dataloader_num_workers": DATALOADER_NUM_WORKERS,
        "dataloader_pin_memory": DATALOADER_PIN_MEMORY,
        "disable_tqdm": DISABLE_TQDM,
    }

def print_config_summary():
    """Print a summary of the current configuration."""
    print("=" * 70)
    print("🚗 CARS & AUTOMOTIVE EXPERT ASSISTANT - CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"\n📦 MODEL:")
    print(f"   Base Model: {MODEL_NAME}")
    print(f"   Quantization: {'4-bit' if USE_4BIT_QUANTIZATION else 'None'}")
    print(f"   Device: {DEVICE}")
    
    print(f"\n🔧 LORA:")
    print(f"   Rank (r): {LORA_R}")
    print(f"   Alpha: {LORA_ALPHA}")
    print(f"   Dropout: {LORA_DROPOUT}")
    print(f"   Target Modules: {', '.join(LORA_TARGET_MODULES)}")
    
    print(f"\n🎓 TRAINING:")
    print(f"   Epochs: {NUM_TRAIN_EPOCHS}")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Batch Size: {PER_DEVICE_TRAIN_BATCH_SIZE} (effective: {PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"   Max Sequence Length: {MAX_SEQ_LENGTH}")
    print(f"   Optimizer: {OPTIM}")
    
    print(f"\n📊 DATASET:")
    print(f"   Training Examples: {NUM_TRAINING_EXAMPLES}")
    print(f"   Train/Test Split: {TRAIN_TEST_SPLIT:.0%}")
    
    print(f"\n💾 OUTPUTS:")
    print(f"   Model Save Directory: {MODEL_SAVE_DIR}")
    print(f"   Results Directory: {OUTPUT_DIR}")
    
    print("=" * 70)
    print()

if __name__ == "__main__":
    print_config_summary()
    validate_config()
