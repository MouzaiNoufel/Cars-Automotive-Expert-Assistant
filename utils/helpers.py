"""
Helper Utilities
Common utility functions for the project.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import torch
import config


def set_seed(seed: int = None):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed (uses config.SEED if None)
    """
    if seed is None:
        seed = config.SEED
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"✓ Random seed set to {seed} for reproducibility")


def print_gpu_info():
    """Print GPU information if available."""
    if torch.cuda.is_available():
        print("\n" + "="*70)
        print("🖥️  GPU INFORMATION")
        print("="*70)
        
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")
        
        for i in range(num_gpus):
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / 1024**3
            print(f"  Total Memory: {total_memory:.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            
            if i == torch.cuda.current_device():
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"  Current Allocated: {allocated:.2f} GB")
                print(f"  Current Reserved: {reserved:.2f} GB")
                print(f"  Current Free: {total_memory - reserved:.2f} GB")
        
        print(f"\nCUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print("="*70 + "\n")
    else:
        print("\n⚠️  No GPU detected. Training will use CPU (very slow).")
        print("   Consider using a CUDA-capable GPU for reasonable training times.\n")


def check_system_requirements():
    """Check if system meets requirements for training."""
    print("\n" + "="*70)
    print("🔍 SYSTEM REQUIREMENTS CHECK")
    print("="*70)
    
    # Check Python version
    import sys
    python_version = sys.version_info
    print(f"\nPython Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("  ❌ Python 3.8+ required")
    else:
        print("  ✓ Python version OK")
    
    # Check PyTorch
    print(f"\nPyTorch Version: {torch.__version__}")
    print("  ✓ PyTorch installed")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"\nCUDA Available: Yes")
        print(f"  CUDA Version: {torch.version.cuda}")
        print("  ✓ GPU acceleration available")
        
        # Check GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory < 12:
            print(f"  ⚠️  GPU Memory: {gpu_memory:.1f} GB (12+ GB recommended)")
            print("     Training may be slow or run out of memory")
        else:
            print(f"  ✓ GPU Memory: {gpu_memory:.1f} GB")
    else:
        print(f"\nCUDA Available: No")
        print("  ⚠️  No GPU detected - training will be extremely slow")
    
    # Check disk space
    import shutil
    disk_usage = shutil.disk_usage(os.path.dirname(os.path.abspath(__file__)))
    free_space = disk_usage.free / 1024**3
    print(f"\nFree Disk Space: {free_space:.1f} GB")
    
    if free_space < 50:
        print("  ⚠️  Less than 50 GB free (50+ GB recommended)")
    else:
        print("  ✓ Sufficient disk space")
    
    print("="*70 + "\n")


def format_time(seconds: float) -> str:
    """
    Format seconds into readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def count_parameters(model) -> dict:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dict with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params,
        'trainable_percent': 100 * trainable_params / total_params if total_params > 0 else 0
    }


def print_model_info(model):
    """Print detailed model information."""
    print("\n" + "="*70)
    print("🤖 MODEL INFORMATION")
    print("="*70)
    
    # Count parameters
    param_info = count_parameters(model)
    
    print(f"\nParameter Counts:")
    print(f"  Total parameters: {param_info['total']:,}")
    print(f"  Trainable parameters: {param_info['trainable']:,}")
    print(f"  Frozen parameters: {param_info['frozen']:,}")
    print(f"  Trainable percentage: {param_info['trainable_percent']:.4f}%")
    
    # Model size estimate
    param_size_mb = param_info['total'] * 4 / 1024**2  # 4 bytes per float32
    print(f"\nEstimated Model Size: {param_size_mb:.2f} MB (float32)")
    
    print("="*70 + "\n")


def create_output_directories():
    """Create necessary output directories."""
    directories = [
        config.OUTPUT_DIR,
        config.MODEL_SAVE_DIR,
        config.LOGS_DIR,
        config.DATASET_CACHE_DIR,
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✓ Output directories created/verified")


def save_training_summary(output_path: str, **kwargs):
    """
    Save training summary to JSON file.
    
    Args:
        output_path: Path to save summary
        **kwargs: Key-value pairs to save
    """
    import json
    from datetime import datetime
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        **kwargs
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Training summary saved to {output_path}")


def clear_gpu_cache():
    """Clear GPU cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ GPU cache cleared")


def print_banner(text: str, char: str = "="):
    """
    Print a text banner.
    
    Args:
        text: Text to display
        char: Character for border
    """
    width = 70
    print("\n" + char * width)
    print(text.center(width))
    print(char * width + "\n")


if __name__ == "__main__":
    # Test utilities
    print_banner("TESTING UTILITIES")
    
    set_seed(42)
    check_system_requirements()
    print_gpu_info()
    
    print("\n✓ All utility functions working correctly!")
