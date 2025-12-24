"""
Evaluator Module
Tests fine-tuned model on automotive queries.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from typing import List, Dict
import config


class AutomotiveEvaluator:
    """Evaluates automotive expert model on test queries."""
    
    def __init__(self, model, tokenizer):
        """
        Initialize evaluator.
        
        Args:
            model: Fine-tuned model
            tokenizer: Tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = config.DEVICE
        
        # Set generation config
        self.generation_config = config.GENERATION_CONFIG.copy()
        self.generation_config['pad_token_id'] = tokenizer.pad_token_id
        self.generation_config['eos_token_id'] = tokenizer.eos_token_id
        
        # Set model to eval mode
        self.model.eval()
    
    def format_prompt_for_inference(self, instruction: str, input_text: str = "") -> str:
        """
        Format instruction for inference (without the output).
        
        Args:
            instruction: The instruction/question
            input_text: Additional context (optional)
            
        Returns:
            Formatted prompt
        """
        if input_text and input_text.strip():
            prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
        else:
            prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""
        
        return prompt
    
    def generate_response(self, instruction: str, input_text: str = "") -> str:
        """
        Generate response for a given instruction.
        
        Args:
            instruction: User instruction/question
            input_text: Additional context
            
        Returns:
            Generated response
        """
        # Format prompt
        prompt = self.format_prompt_for_inference(instruction, input_text)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH
        )
        
        # Move to device
        if torch.cuda.is_available():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self.generation_config
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the response (after "### Response:")
        if "### Response:" in generated_text:
            response = generated_text.split("### Response:")[-1].strip()
        else:
            response = generated_text
        
        return response
    
    def evaluate_on_queries(self, queries: List[Dict[str, str]]) -> List[Dict]:
        """
        Evaluate model on a list of queries.
        
        Args:
            queries: List of dicts with 'instruction' and optional 'input'
            
        Returns:
            List of results with query and response
        """
        results = []
        
        print(f"\n{'='*70}")
        print(f"🔍 EVALUATING ON {len(queries)} QUERIES")
        print(f"{'='*70}\n")
        
        for i, query in enumerate(queries, 1):
            instruction = query['instruction']
            input_text = query.get('input', '')
            
            print(f"Query {i}/{len(queries)}:")
            print(f"Q: {instruction}")
            if input_text:
                print(f"Context: {input_text}")
            
            # Generate response
            response = self.generate_response(instruction, input_text)
            
            print(f"A: {response}\n")
            print("-" * 70 + "\n")
            
            results.append({
                'instruction': instruction,
                'input': input_text,
                'response': response
            })
        
        return results
    
    def run_standard_evaluation(self):
        """
        Run evaluation on standard automotive queries from config.
        
        Returns:
            Evaluation results
        """
        print("\n" + "="*70)
        print("🚗 AUTOMOTIVE EXPERT EVALUATION")
        print("="*70)
        print("\nTesting model on automotive domain questions...")
        
        results = self.evaluate_on_queries(config.EVALUATION_QUERIES)
        
        print("="*70)
        print("✓ EVALUATION COMPLETE")
        print("="*70)
        
        return results
    
    def interactive_test(self):
        """Run interactive testing session."""
        print("\n" + "="*70)
        print("💬 INTERACTIVE AUTOMOTIVE EXPERT TESTING")
        print("="*70)
        print("\nAsk me anything about cars! (Type 'quit' to exit)\n")
        
        while True:
            try:
                # Get user input
                instruction = input("Your question: ").strip()
                
                if not instruction:
                    continue
                
                if instruction.lower() in ['quit', 'exit', 'q']:
                    print("\nExiting interactive mode. Goodbye!")
                    break
                
                # Optional context
                context = input("Additional context (optional, press Enter to skip): ").strip()
                
                print("\nGenerating response...\n")
                
                # Generate response
                response = self.generate_response(instruction, context)
                
                print(f"🚗 Assistant: {response}\n")
                print("-" * 70 + "\n")
                
            except KeyboardInterrupt:
                print("\n\nExiting interactive mode. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


def evaluate_model(model, tokenizer, queries: List[Dict[str, str]] = None):
    """
    Convenience function to evaluate model.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        queries: Optional custom queries (uses config queries if None)
        
    Returns:
        Evaluation results
    """
    evaluator = AutomotiveEvaluator(model, tokenizer)
    
    if queries is None:
        # Use standard evaluation queries
        return evaluator.run_standard_evaluation()
    else:
        # Use custom queries
        return evaluator.evaluate_on_queries(queries)


if __name__ == "__main__":
    print("This module should be imported, not run directly.")
    print("Use scripts/evaluate.py to evaluate the model.")
