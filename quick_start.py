"""
Quick Start Script
Verifies installation and provides usage instructions.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def check_imports():
    """Check if all required packages are installed."""
    print("\n" + "="*70)
    print("CHECKING DEPENDENCIES")
    print("="*70 + "\n")
    
    missing = []
    
    packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('datasets', 'Datasets'),
        ('peft', 'PEFT'),
        ('bitsandbytes', 'BitsAndBytes'),
        ('accelerate', 'Accelerate'),
    ]
    
    for module, name in packages:
        try:
            __import__(module)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"❌ {name} NOT installed")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"\nInstall with: pip install -r requirements.txt")
        return False
    else:
        print(f"\n✓ All dependencies installed!")
        return True


def print_usage():
    """Print usage instructions."""
    print("\n" + "="*70)
    print("🚗 CARS & AUTOMOTIVE EXPERT ASSISTANT")
    print("="*70)
    
    print("\n📖 QUICK START GUIDE\n")
    
    print("1️⃣  INSTALL DEPENDENCIES")
    print("   pip install -r requirements.txt")
    
    print("\n2️⃣  (OPTIONAL) LOGIN TO HUGGING FACE")
    print("   huggingface-cli login")
    print("   (Required only for Llama models)")
    
    print("\n3️⃣  TRAIN THE MODEL")
    print("   python scripts/train.py")
    print("   • Generates automotive dataset")
    print("   • Downloads and quantizes base model")
    print("   • Fine-tunes with LoRA")
    print("   • Saves to ./automotive_expert_model/")
    print("   • Takes 1-3 hours on modern GPU")
    
    print("\n4️⃣  EVALUATE THE MODEL")
    print("   python scripts/evaluate.py")
    print("   • Tests on automotive questions")
    print("   • Shows model capabilities")
    
    print("\n5️⃣  CHAT WITH YOUR ASSISTANT")
    print("   python scripts/inference.py")
    print("   • Interactive Q&A about cars")
    print("   • Ask anything automotive-related")
    
    print("\n" + "="*70)
    print("⚙️  CONFIGURATION")
    print("="*70)
    
    print("\nEdit config.py to customize:")
    print("  • Base model (Mistral-7B or Llama-2-7B)")
    print("  • LoRA parameters (rank, alpha, dropout)")
    print("  • Training hyperparameters")
    print("  • Dataset size")
    print("  • Generation settings")
    
    print("\n" + "="*70)
    print("📁 PROJECT STRUCTURE")
    print("="*70)
    
    print("""
Cars & Automotive Expert Assistant/
├── README.md              # Full documentation
├── requirements.txt       # Dependencies
├── config.py             # Configuration hub
│
├── data/                 # Dataset generation
│   ├── dataset_generator.py
│   └── preprocessor.py
│
├── models/               # Model loading
│   └── model_loader.py
│
├── training/             # Training logic
│   └── trainer.py
│
├── evaluation/           # Model evaluation
│   └── evaluator.py
│
├── inference/            # Chat interface
│   └── chat.py
│
├── scripts/              # Main scripts
│   ├── train.py         # Train model
│   ├── evaluate.py      # Evaluate model
│   └── inference.py     # Chat interface
│
└── utils/                # Utilities
    └── helpers.py
""")
    
    print("="*70)
    print("💡 TIPS")
    print("="*70)
    
    print("""
• GPU Requirement: 12GB+ VRAM (RTX 3090, 4090, or better)
• First run downloads ~13GB model (cached for future use)
• Training generates ~500 automotive Q&A examples
• Model learns car specs, comparisons, buying advice, etc.
• LoRA adapters are only ~40MB (easy to share/version)
• TensorBoard logs: tensorboard --logdir logs/
""")
    
    print("="*70)
    print("🎯 WHAT THIS PROJECT DOES")
    print("="*70)
    
    print("""
This project fine-tunes a 7B parameter language model to become
an expert automotive assistant using:

✓ Parameter-Efficient Fine-Tuning (LoRA)
✓ 4-bit Quantization (QLoRA) for memory efficiency
✓ Domain-specific automotive dataset (500+ examples)
✓ Professional ML engineering practices

The result: A specialized AI that can:
• Explain car technologies (engines, hybrids, EVs)
• Compare vehicles (performance, reliability, value)
• Give buying advice (budget, needs, maintenance)
• Answer maintenance questions
• Discuss automotive trends

Perfect for:
• Automotive dealerships (customer support)
• Car review platforms (content generation)
• Educational tools (learning about cars)
• Portfolio projects (demonstrate ML skills)
""")
    
    print("="*70)
    print("📚 RESOURCES")
    print("="*70)
    
    print("""
• README.md - Complete project documentation
• config.py - All settings and hyperparameters
• HuggingFace Docs - https://huggingface.co/docs
• LoRA Paper - https://arxiv.org/abs/2106.09685
• QLoRA Paper - https://arxiv.org/abs/2305.14314
""")
    
    print("="*70)
    print("🚀 READY TO START!")
    print("="*70)
    
    print("""
If dependencies are installed, begin training:

    python scripts/train.py

Questions? Check README.md for detailed documentation.
""")
    
    print("="*70 + "\n")


def main():
    """Main quick start function."""
    
    print("\n" + "="*70)
    print("🚗 AUTOMOTIVE EXPERT ASSISTANT - QUICK START")
    print("="*70)
    
    # Check dependencies
    deps_ok = check_imports()
    
    # Print usage
    print_usage()
    
    if not deps_ok:
        print("⚠️  Please install missing dependencies first:")
        print("   pip install -r requirements.txt\n")
    else:
        print("✅ System ready! You can start training:")
        print("   python scripts/train.py\n")


if __name__ == "__main__":
    main()
