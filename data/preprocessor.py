"""
Dataset Preprocessor
Formats automotive dataset for training with proper tokenization.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List
from datasets import Dataset
from transformers import AutoTokenizer
import config


class DatasetPreprocessor:
    """Preprocesses automotive dataset for LLM fine-tuning."""
    
    def __init__(self, tokenizer: AutoTokenizer):
        """
        Initialize preprocessor with tokenizer.
        
        Args:
            tokenizer: HuggingFace tokenizer for the model
        """
        self.tokenizer = tokenizer
        self.max_length = config.MAX_SEQ_LENGTH
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def format_instruction(self, example: Dict[str, str]) -> str:
        """
        Format a single example into instruction-following prompt.
        
        Args:
            example: Dict with 'instruction', 'input', 'output' keys
            
        Returns:
            Formatted prompt string
        """
        instruction = example['instruction']
        input_text = example.get('input', '')
        output = example['output']
        
        # Use appropriate template based on whether input is present
        if input_text and input_text.strip():
            prompt = config.PROMPT_TEMPLATE.format(
                instruction=instruction,
                input=input_text,
                output=output
            )
        else:
            prompt = config.PROMPT_TEMPLATE_NO_INPUT.format(
                instruction=instruction,
                output=output
            )
        
        return prompt
    
    def tokenize_function(self, examples: Dict[str, List]) -> Dict:
        """
        Tokenize a batch of examples.
        
        Args:
            examples: Batch of examples from dataset
            
        Returns:
            Tokenized batch with input_ids, attention_mask, labels
        """
        # Format all examples in batch
        prompts = [
            self.format_instruction({
                'instruction': inst,
                'input': inp,
                'output': out
            })
            for inst, inp, out in zip(
                examples['instruction'],
                examples['input'],
                examples['output']
            )
        ]
        
        # Tokenize with padding and truncation
        model_inputs = self.tokenizer(
            prompts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors=None,  # Return lists, not tensors
        )
        
        # For causal LM, labels are the same as input_ids
        # The model will learn to predict next token
        model_inputs['labels'] = model_inputs['input_ids'].copy()
        
        return model_inputs
    
    def prepare_dataset(self, data: List[Dict[str, str]], shuffle: bool = True) -> Dataset:
        """
        Convert raw data to HuggingFace Dataset with tokenization.
        
        Args:
            data: List of examples with instruction/input/output
            shuffle: Whether to shuffle the dataset
            
        Returns:
            Tokenized HuggingFace Dataset
        """
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(data)
        
        if shuffle:
            dataset = dataset.shuffle(seed=config.SEED)
        
        # Apply tokenization
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,  # Remove original text columns
            desc="Tokenizing dataset",
        )
        
        return tokenized_dataset
    
    def split_dataset(self, dataset: Dataset, train_ratio: float = 0.9) -> Dict[str, Dataset]:
        """
        Split dataset into train and validation sets.
        
        Args:
            dataset: Full tokenized dataset
            train_ratio: Proportion for training (rest is validation)
            
        Returns:
            Dict with 'train' and 'validation' datasets
        """
        split_dataset = dataset.train_test_split(
            test_size=1.0 - train_ratio,
            seed=config.SEED
        )
        
        return {
            'train': split_dataset['train'],
            'validation': split_dataset['test']
        }
    
    def get_sample_prompt(self, example: Dict[str, str]) -> str:
        """
        Get formatted prompt for a single example (for testing).
        
        Args:
            example: Dict with instruction/input/output
            
        Returns:
            Formatted prompt string
        """
        return self.format_instruction(example)
    
    def print_dataset_stats(self, dataset: Dataset, name: str = "Dataset"):
        """Print statistics about the dataset."""
        print(f"\n{'='*70}")
        print(f"{name.upper()} STATISTICS")
        print(f"{'='*70}")
        print(f"Number of examples: {len(dataset)}")
        
        if len(dataset) > 0:
            # Calculate average sequence length
            sample = dataset[0]
            if 'input_ids' in sample:
                lengths = [len(example['input_ids']) for example in dataset]
                avg_length = sum(lengths) / len(lengths)
                max_length = max(lengths)
                min_length = min(lengths)
                
                print(f"Average sequence length: {avg_length:.1f} tokens")
                print(f"Max sequence length: {max_length} tokens")
                print(f"Min sequence length: {min_length} tokens")
            
            print(f"\nSample tokenized sequence:")
            print(f"Input IDs length: {len(sample['input_ids'])}")
            print(f"First 10 tokens: {sample['input_ids'][:10]}")
        
        print(f"{'='*70}\n")


def create_training_datasets(raw_data: List[Dict[str, str]], tokenizer: AutoTokenizer):
    """
    Convenience function to create train/validation datasets from raw data.
    
    Args:
        raw_data: List of examples
        tokenizer: Model tokenizer
        
    Returns:
        Tuple of (train_dataset, validation_dataset, preprocessor)
    """
    preprocessor = DatasetPreprocessor(tokenizer)
    
    # Prepare and tokenize full dataset
    print(f"📊 Preparing dataset with {len(raw_data)} examples...")
    full_dataset = preprocessor.prepare_dataset(
        raw_data,
        shuffle=config.SHUFFLE_DATASET
    )
    
    # Split into train/validation
    datasets = preprocessor.split_dataset(
        full_dataset,
        train_ratio=config.TRAIN_TEST_SPLIT
    )
    
    train_dataset = datasets['train']
    val_dataset = datasets['validation']
    
    # Print statistics
    preprocessor.print_dataset_stats(train_dataset, "Training Set")
    preprocessor.print_dataset_stats(val_dataset, "Validation Set")
    
    # Show sample formatted prompt
    print(f"{'='*70}")
    print("SAMPLE FORMATTED PROMPT")
    print(f"{'='*70}")
    sample_example = raw_data[0]
    sample_prompt = preprocessor.get_sample_prompt(sample_example)
    print(sample_prompt[:500] + "..." if len(sample_prompt) > 500 else sample_prompt)
    print(f"{'='*70}\n")
    
    return train_dataset, val_dataset, preprocessor


if __name__ == "__main__":
    # Test with sample data
    from data.dataset_generator import AutomotiveDatasetGenerator
    
    print("Generating sample automotive dataset...")
    generator = AutomotiveDatasetGenerator(seed=42)
    raw_data = generator.generate_dataset(num_examples=100)
    
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    print("\nPreprocessing dataset...")
    train_ds, val_ds, preprocessor = create_training_datasets(raw_data, tokenizer)
    
    print(f"\n✓ Successfully created datasets!")
    print(f"  Training examples: {len(train_ds)}")
    print(f"  Validation examples: {len(val_ds)}")
