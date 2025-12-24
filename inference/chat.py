"""
Chat Interface
Interactive chat with automotive expert assistant.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import config


class AutomotiveChat:
    """Interactive chat interface for automotive expert."""
    
    def __init__(self, model, tokenizer):
        """
        Initialize chat interface.
        
        Args:
            model: Fine-tuned automotive expert model
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
        
        # Conversation history (optional for future context awareness)
        self.history = []
    
    def format_prompt(self, question: str, context: str = "") -> str:
        """
        Format user question into prompt.
        
        Args:
            question: User's question
            context: Optional context
            
        Returns:
            Formatted prompt
        """
        if context and context.strip():
            prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{question}

### Input:
{context}

### Response:
"""
        else:
            prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{question}

### Response:
"""
        
        return prompt
    
    def generate_response(self, question: str, context: str = "") -> str:
        """
        Generate response to user question.
        
        Args:
            question: User question
            context: Optional context
            
        Returns:
            Generated response
        """
        # Format prompt
        prompt = self.format_prompt(question, context)
        
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
        
        # Extract response
        if "### Response:" in generated_text:
            response = generated_text.split("### Response:")[-1].strip()
        else:
            response = generated_text
        
        return response
    
    def print_welcome(self):
        """Print welcome message."""
        print("\n" + "="*70)
        print("🚗 AUTOMOTIVE EXPERT ASSISTANT")
        print("="*70)
        print("\nWelcome! I'm your automotive expert assistant.")
        print("I can help you with:")
        print("  • Car specifications and technologies")
        print("  • Vehicle comparisons and recommendations")
        print("  • Buying advice and budgeting")
        print("  • Maintenance and repair guidance")
        print("  • Engine types (ICE, Hybrid, Electric)")
        print("  • Safety features and ratings")
        print("  • And much more about cars!")
        print("\nCommands:")
        print("  • Type your question and press Enter")
        print("  • Type 'help' for example questions")
        print("  • Type 'quit' or 'exit' to end chat")
        print("="*70 + "\n")
    
    def print_help(self):
        """Print example questions."""
        print("\n" + "="*70)
        print("EXAMPLE QUESTIONS")
        print("="*70)
        examples = [
            "What's the difference between AWD and 4WD?",
            "Compare Toyota Camry and Honda Accord",
            "Best used car under $15,000?",
            "How does a hybrid car work?",
            "Should I buy new or used?",
            "What are the most important safety features?",
            "Explain turbocharging",
            "How often should I change my oil?",
            "What is regenerative braking?",
            "Diesel vs gasoline for a truck?",
        ]
        
        for i, example in enumerate(examples, 1):
            print(f"{i:2}. {example}")
        
        print("="*70 + "\n")
    
    def run(self):
        """Run interactive chat session."""
        self.print_welcome()
        
        while True:
            try:
                # Get user input
                user_input = input("🙋 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thanks for chatting! Drive safe!")
                    break
                
                if user_input.lower() in ['help', 'h', '?']:
                    self.print_help()
                    continue
                
                # Generate response
                print("\n🤔 Thinking...\n")
                response = self.generate_response(user_input)
                
                # Print response
                print(f"🚗 Assistant: {response}\n")
                print("-" * 70 + "\n")
                
                # Store in history
                self.history.append({
                    'user': user_input,
                    'assistant': response
                })
                
            except KeyboardInterrupt:
                print("\n\n👋 Thanks for chatting! Drive safe!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try asking your question differently.\n")
    
    def single_query(self, question: str, context: str = "") -> str:
        """
        Get single response without interactive mode.
        
        Args:
            question: User question
            context: Optional context
            
        Returns:
            Generated response
        """
        response = self.generate_response(question, context)
        return response


def start_chat(model, tokenizer):
    """
    Start interactive chat session.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
    """
    chat = AutomotiveChat(model, tokenizer)
    chat.run()


if __name__ == "__main__":
    print("This module should be imported, not run directly.")
    print("Use scripts/inference.py to start the chat interface.")
