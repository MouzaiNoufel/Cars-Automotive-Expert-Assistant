# Cars & Automotive Expert Assistant - LLM Fine-Tuning Project

## 🚗 Project Overview

This is a production-ready machine learning project that fine-tunes an open-source Large Language Model (Mistral-7B or Llama-2-7B) to become a specialized **Automotive Domain Expert Assistant**. The model is trained to provide expert-level knowledge on car specifications, technologies, buying advice, maintenance, and automotive engineering concepts.

**Key Capabilities:**
- Explain car specifications and automotive technologies
- Compare vehicles across performance, comfort, reliability metrics
- Provide personalized buying advice based on budget, usage, and needs
- Explain complex automotive systems (engines, transmissions, hybrids, EVs)
- Answer questions about tuning, safety features, and fuel efficiency
- Discuss maintenance schedules and common repair issues

## 🎯 Why This Project Matters

### Business Value
- **Automotive Dealerships**: Automated customer support and product recommendations
- **Car Review Platforms**: Generate detailed technical explanations and comparisons
- **Educational Platforms**: Teach automotive concepts to enthusiasts and students
- **Insurance Companies**: Assess vehicle specifications and risk factors
- **Fleet Management**: Assist in vehicle selection and maintenance planning

### Technical Excellence
- **Parameter-Efficient Fine-Tuning (PEFT)**: Uses LoRA/QLoRA for memory-efficient training
- **4-bit Quantization**: Enables training on consumer GPUs (12-16GB VRAM)
- **Domain Specialization**: Focused dataset with automotive-specific instruction-following examples
- **Production-Ready**: Clean architecture, error handling, reproducibility, and logging

## 🛠️ Technical Architecture

### Base Model Selection
- **Primary**: `mistralai/Mistral-7B-v0.1` (recommended for quality/efficiency balance)
- **Alternative**: `meta-llama/Llama-2-7b-hf`

Both models offer:
- Strong baseline instruction-following capabilities
- Efficient inference on consumer hardware
- Active community support

### Fine-Tuning Strategy: LoRA + QLoRA

**Why LoRA (Low-Rank Adaptation)?**
- Adds trainable low-rank matrices to attention layers
- Reduces trainable parameters from 7B to ~10-20M (99.7% reduction)
- Preserves base model knowledge while injecting domain expertise
- Enables fast experimentation and deployment

**Why QLoRA (Quantized LoRA)?**
- Loads base model in 4-bit precision using bitsandbytes
- Reduces VRAM from ~28GB to ~6-8GB for 7B models
- Maintains training quality through careful quantization
- Makes fine-tuning accessible on RTX 3090, 4090, or similar GPUs

**Quantization Configuration:**
- 4-bit NormalFloat (NF4) quantization
- Double quantization for additional memory savings
- BF16 compute dtype for stable training
- Nested quantization enabled

### Training Configuration
- **Learning Rate**: 2e-4 (optimized for LoRA)
- **Batch Size**: 4 per device with gradient accumulation
- **LoRA Rank**: 64 (balance between capacity and efficiency)
- **LoRA Alpha**: 16 (scaling factor)
- **Target Modules**: q_proj, k_proj, v_proj, o_proj (attention layers)
- **Optimizer**: AdamW with 8-bit precision
- **Scheduler**: Linear warmup with cosine decay

## 📊 Dataset Design

The training dataset is specifically designed for automotive domain expertise:

**Dataset Structure:**
```json
{
  "instruction": "User's question or task",
  "input": "Additional context (optional)",
  "output": "Expert automotive response"
}
```

**Topic Coverage:**
1. **Vehicle Specifications**: Engine size, horsepower, torque, dimensions
2. **Powertrain Technologies**: ICE, Hybrid, PHEV, BEV architectures
3. **Performance Metrics**: 0-60 times, top speed, handling characteristics
4. **Safety Features**: ADAS, crash ratings, active/passive safety
5. **Maintenance**: Service intervals, common issues, cost of ownership
6. **Buying Guidance**: Budget recommendations, use-case matching
7. **Automotive Engineering**: How systems work (braking, suspension, drivetrain)
8. **Electric Vehicles**: Battery technology, charging, range considerations
9. **Comparisons**: Head-to-head vehicle analysis
10. **Market Trends**: Automotive industry developments

**Dataset Quality:**
- 500+ high-quality instruction-response pairs
- Diverse question formulations
- Accurate technical information
- Natural conversational tone
- Varied complexity levels

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU with 12GB+ VRAM (recommended)
- 50GB+ free disk space

### Step 1: Clone or Download Project
```bash
cd "c:\Users\pc\Desktop\ai projects\Cars & Automotive Expert Assistant"
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure Hugging Face Access
```bash
# Login to Hugging Face (required for Llama models)
huggingface-cli login
```

## 🎓 Training the Model

### Quick Start
```bash
python scripts/train.py
```

### Training Process
1. **Dataset Generation**: Creates automotive instruction dataset
2. **Model Loading**: Downloads and quantizes base model (4-bit)
3. **LoRA Setup**: Configures trainable adapter layers
4. **Training**: Fine-tunes on automotive data with progress tracking
5. **Checkpointing**: Saves model every 50 steps
6. **Evaluation**: Tests on validation set

### Expected Training Time
- **RTX 3090 (24GB)**: ~2-3 hours for 3 epochs
- **RTX 4090 (24GB)**: ~1.5-2 hours for 3 epochs
- **A100 (40GB)**: ~1 hour for 3 epochs

### Training Output
- Checkpoints saved to `./results/checkpoint-{step}`
- Final model saved to `./automotive_expert_model`
- Training logs in `./results/trainer_log.txt`

## 🔍 Evaluation

### Run Evaluation Script
```bash
python scripts/evaluate.py
```

### What It Tests
- Technical accuracy on car specifications
- Buying recommendation quality
- Maintenance advice relevance
- Comparison analysis depth
- Electric vehicle knowledge
- Engine technology explanations

### Sample Evaluation Questions
- "What's the difference between a turbocharged and supercharged engine?"
- "Compare the Toyota Camry and Honda Accord for a family"
- "Should I buy a hybrid or electric car for daily commuting?"
- "Explain how regenerative braking works"
- "What maintenance does a Tesla Model 3 need?"

## 💬 Inference & Chat

### Interactive Chat Mode
```bash
python scripts/inference.py
```

### Example Interactions

**User:** "What are the key differences between AWD and 4WD?"

**Assistant:** "AWD (All-Wheel Drive) and 4WD (Four-Wheel Drive) both send power to all four wheels, but they work differently:

**AWD**: Automatic system that distributes power as needed. Better for on-road driving, improved traction in rain/snow. Found in crossovers and sedans like Subaru Outback.

**4WD**: Manual engagement, typically has low-range gearing for off-road. Used in trucks and SUVs like Jeep Wrangler. Better for serious off-roading, towing, and extreme conditions.

Choose AWD for daily driving with occasional bad weather, 4WD for off-road adventures and heavy-duty work."

---

**User:** "Best used car under $15,000?"

**Assistant:** "For $15,000, I recommend these reliable options:

1. **Honda Civic (2015-2017)**: Excellent reliability, 30+ MPG, low maintenance costs
2. **Toyota Corolla (2014-2016)**: Best-in-class dependability, great resale value
3. **Mazda3 (2016-2018)**: Fun to drive, upscale interior, good fuel economy
4. **Honda CR-V (2012-2014)**: Practical SUV, spacious, Honda reliability

**Key Tips:**
- Get pre-purchase inspection
- Check CARFAX for accidents
- Look for one-owner vehicles
- Verify maintenance records
- Budget $500-1000 for immediate repairs

All these have strong reliability ratings and affordable parts."

## 📁 Project Structure

```
Cars & Automotive Expert Assistant/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config.py                    # Central configuration
│
├── data/
│   ├── __init__.py
│   ├── dataset_generator.py    # Generates automotive dataset
│   └── preprocessor.py          # Dataset formatting & tokenization
│
├── models/
│   ├── __init__.py
│   └── model_loader.py          # Model loading with quantization
│
├── training/
│   ├── __init__.py
│   └── trainer.py               # Training logic with LoRA
│
├── evaluation/
│   ├── __init__.py
│   └── evaluator.py             # Model evaluation
│
├── inference/
│   ├── __init__.py
│   └── chat.py                  # Interactive chat interface
│
├── scripts/
│   ├── train.py                 # Main training script
│   ├── evaluate.py              # Evaluation runner
│   └── inference.py             # Chat runner
│
└── utils/
    ├── __init__.py
    └── helpers.py               # Utility functions
```

## 🔧 Configuration

All hyperparameters are centralized in `config.py`:

- **Model Settings**: Base model selection, quantization config
- **LoRA Settings**: Rank, alpha, dropout, target modules
- **Training Settings**: Learning rate, batch size, epochs
- **Dataset Settings**: Size, split ratios
- **Paths**: Output directories, checkpoint locations

Modify `config.py` to experiment with different settings.

## 🎯 Why This Approach Works

### 1. **Memory Efficiency**
QLoRA enables training 7B models on consumer GPUs by:
- 4-bit quantization reduces model size by 75%
- Only training 0.3% of parameters (LoRA adapters)
- Gradient checkpointing for further memory savings

### 2. **Domain Specialization**
Fine-tuning on automotive-specific data:
- Teaches technical terminology and concepts
- Improves factual accuracy on car specifications
- Develops conversational patterns for buying advice
- Maintains general language understanding from base model

### 3. **Fast Iteration**
LoRA adapters are small (~40MB) compared to full models (13GB+):
- Quick to train (hours vs. days)
- Easy to version and swap
- Enable A/B testing different approaches
- Lower storage and deployment costs

### 4. **Production Readiness**
- Reproducible with fixed random seeds
- Error handling for GPU/CPU scenarios
- Checkpointing for recovery from failures
- Clean code architecture for maintenance

## 📈 Expected Results

After fine-tuning, you should observe:
- **Improved Technical Accuracy**: Better responses on automotive terminology
- **Domain-Specific Knowledge**: Detailed explanations of car systems
- **Practical Advice**: Actionable buying and maintenance recommendations
- **Conversational Quality**: Natural dialogue about cars

## 🔄 Next Steps & Improvements

### Dataset Enhancement
- Integrate real car review data (Edmunds, Car and Driver)
- Add multilingual automotive content
- Include historical car information
- Expand to motorcycles, commercial vehicles

### Model Improvements
- Experiment with larger models (13B, 70B with multi-GPU)
- Try different LoRA configurations (higher rank)
- Implement DPO/RLHF for preference alignment
- Add retrieval-augmented generation (RAG) for current specs

### Deployment
- Build REST API with FastAPI
- Create web interface for chat
- Deploy on cloud (AWS SageMaker, Google Cloud)
- Mobile app integration

### Evaluation
- Human evaluation with automotive experts
- A/B testing against general-purpose models
- Benchmark on automotive Q&A datasets
- Track user satisfaction metrics

## 📚 References & Resources

### Papers
- **LoRA**: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- **QLoRA**: "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)
- **Instruction Tuning**: "Finetuned Language Models Are Zero-Shot Learners" (Wei et al., 2021)

### Libraries
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT](https://huggingface.co/docs/peft)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)

### Automotive Data Sources
- NHTSA Vehicle Safety Data
- EPA Fuel Economy Data
- Edmunds API
- Car and Driver Reviews

## 👥 Target Audience

This project is valuable for:
- **ML Engineers**: Learn LLM fine-tuning best practices
- **Automotive Startups**: Build AI-powered car advisors
- **Dealerships**: Automate customer support
- **Educational Platforms**: Create interactive learning tools
- **Portfolio Projects**: Demonstrate end-to-end ML skills

## 📄 License

This project is for educational and commercial use. Base models have their own licenses (Apache 2.0 for Mistral, custom for Llama-2).

## 🤝 Contributing

To improve this project:
1. Expand the automotive dataset with verified information
2. Add support for more base models
3. Implement additional evaluation metrics
4. Optimize training hyperparameters
5. Create deployment guides

## ⚠️ Disclaimer

This model provides general automotive information and advice. Always:
- Consult certified mechanics for repairs
- Verify specifications with manufacturers
- Get professional inspections before purchases
- Follow official maintenance schedules

---

**Built with ❤️ for automotive enthusiasts and AI engineers**

*This project demonstrates production-ready LLM fine-tuning using modern techniques (LoRA, quantization) applied to a valuable real-world domain (automotive expertise).*
