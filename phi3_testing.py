import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import time
import json
import os
from datetime import datetime
import argparse

# --- Configuration ---
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

# Generation parameters (can be adjusted)
DEFAULT_MAX_NEW_TOKENS = 200
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_DO_SAMPLE = True
DEFAULT_REPETITION_PENALTY = 1.1

class Phi3ChatBot:
    def __init__(self, model_name=MODEL_NAME, save_history=False):
        self.model_name = model_name
        self.save_history = save_history
        self.session_start = datetime.now()
        
        # Generation parameters
        self.max_new_tokens = DEFAULT_MAX_NEW_TOKENS
        self.temperature = DEFAULT_TEMPERATURE
        self.top_p = DEFAULT_TOP_P
        self.do_sample = DEFAULT_DO_SAMPLE
        self.repetition_penalty = DEFAULT_REPETITION_PENALTY
        
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize conversation
        self.current_messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
        self.conversation_history = []
        
        # Load model and tokenizer
        self._load_model()
        
    def _load_model(self):
        """Load the model and tokenizer with error handling."""
        print(f"\nLoading model: {self.model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Set padding token - use a different token than eos to avoid attention mask issues
            if self.tokenizer.pad_token is None:
                if self.tokenizer.unk_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.unk_token
                else:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    # Also set pad_token_id explicitly
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto",
                # Disable problematic caching
                use_cache=False
            )
            self.model.eval()
            print("Model loaded successfully!")
            
            # Skip pipeline due to compatibility issues with Phi-3
            print("Skipping pipeline initialization due to known compatibility issues with Phi-3.")
            print("Using manual tokenization for better compatibility.")
            self.pipe = None
                
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please check the model name, your internet connection, or accept terms on Hugging Face.")
            raise
    
    def _save_conversation(self):
        """Save conversation history to a JSON file."""
        if not self.save_history:
            return
            
        filename = f"phi3_chat_{self.session_start.strftime('%Y%m%d_%H%M%S')}.json"
        conversation_data = {
            "session_start": self.session_start.isoformat(),
            "model_name": self.model_name,
            "conversation": self.conversation_history
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            print(f"Conversation saved to {filename}")
        except Exception as e:
            print(f"Error saving conversation: {e}")
    
    def _update_parameters(self, param_string):
        """Update generation parameters from user input."""
        parts = param_string.split(' ')
        for part in parts[1:]:
            try:
                if part.startswith('max_new_tokens='):
                    self.max_new_tokens = int(part.split('=')[1])
                    print(f"Set max_new_tokens to {self.max_new_tokens}")
                elif part.startswith('temp=') or part.startswith('temperature='):
                    self.temperature = float(part.split('=')[1])
                    print(f"Set temperature to {self.temperature}")
                elif part.startswith('top_p='):
                    self.top_p = float(part.split('=')[1])
                    print(f"Set top_p to {self.top_p}")
                elif part.startswith('rep_penalty=') or part.startswith('repetition_penalty='):
                    self.repetition_penalty = float(part.split('=')[1])
                    print(f"Set repetition_penalty to {self.repetition_penalty}")
                elif part.startswith('do_sample='):
                    self.do_sample = part.split('=')[1].lower() in ['true', '1', 'yes']
                    print(f"Set do_sample to {self.do_sample}")
            except (ValueError, IndexError) as e:
                print(f"Invalid parameter format for '{part}': {e}")
    
    def _show_parameters(self):
        """Display current generation parameters."""
        print(f"\nCurrent Generation Parameters:")
        print(f"  max_new_tokens: {self.max_new_tokens}")
        print(f"  temperature: {self.temperature}")
        print(f"  top_p: {self.top_p}")
        print(f"  do_sample: {self.do_sample}")
        print(f"  repetition_penalty: {self.repetition_penalty}")
    
    def _show_help(self):
        """Display help information."""
        print("\n--- Help ---")
        print("Commands:")
        print("  Type your message and press Enter to chat")
        print("  'quit' or 'exit': Stop the session")
        print("  'reset': Clear the conversation history")
        print("  'save': Save current conversation to file")
        print("  'show_history': Display conversation history")
        print("  'show_params': Display current generation parameters")
        print("  'params <param>=<value>': Change generation parameters")
        print("    Available parameters:")
        print("      - max_new_tokens=<int>: Maximum tokens to generate")
        print("      - temp=<float> or temperature=<float>: Sampling temperature")
        print("      - top_p=<float>: Top-p (nucleus) sampling")
        print("      - rep_penalty=<float>: Repetition penalty")
        print("      - do_sample=<true/false>: Enable/disable sampling")
        print("    Example: params max_new_tokens=100 temp=0.5 top_p=0.8")
        print("  'system <message>': Change system prompt")
        print("  'help': Show this help message")
        print("----------------")
    
    def _show_history(self):
        """Display conversation history."""
        if not self.conversation_history:
            print("No conversation history yet.")
            return
            
        print("\n--- Conversation History ---")
        for i, exchange in enumerate(self.conversation_history, 1):
            print(f"{i}. User: {exchange['user']}")
            print(f"   Bot:  {exchange['assistant']}")
            print()
    
    def _set_system_prompt(self, system_message):
        """Update the system prompt."""
        self.current_messages[0] = {"role": "system", "content": system_message}
        print(f"System prompt updated to: {system_message}")
    
    def _generate_response_pipeline(self, messages):
        """Generate response using the pipeline."""
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        # Add temperature and top_p only if do_sample is True
        if self.do_sample:
            generation_kwargs.update({
                "temperature": self.temperature,
                "top_p": self.top_p,
                "repetition_penalty": self.repetition_penalty,
            })
        
        response = self.pipe(messages, **generation_kwargs)
        
        generated_text_full = response[0]['generated_text']
        
        # Extract assistant's response
        assistant_marker = "<|assistant|>"
        end_marker = "<|end|>"
        
        last_assistant_index = generated_text_full.rfind(assistant_marker)
        if last_assistant_index != -1:
            assistant_response_part = generated_text_full[last_assistant_index + len(assistant_marker):]
            end_index = assistant_response_part.find(end_marker)
            if end_index != -1:
                assistant_reply = assistant_response_part[:end_index].strip()
            else:
                assistant_reply = assistant_response_part.strip()
        else:
            # Fallback: try to extract the last message
            if isinstance(generated_text_full, list) and len(generated_text_full) > len(messages):
                assistant_reply = generated_text_full[-1].get('content', '').strip()
            else:
                assistant_reply = str(generated_text_full).strip()
        
        return assistant_reply
    
    def _generate_response_manual(self, messages):
        """Generate response using manual tokenization with compatibility fixes."""
        try:
            # Apply chat template
            encoded_chat = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Create explicit attention mask to avoid the warning
            # Make sure pad tokens are properly masked
            if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                # If pad and eos are the same, create a mask that attends to all tokens
                attention_mask = torch.ones_like(encoded_chat)
            else:
                attention_mask = encoded_chat.ne(self.tokenizer.pad_token_id).long()
            
            # Prepare generation kwargs with compatibility fixes
            generation_kwargs = {
                "input_ids": encoded_chat,
                "attention_mask": attention_mask,
                "max_new_tokens": min(self.max_new_tokens, 100),  # Limit to avoid issues
                "do_sample": self.do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": False,  # Disable cache to avoid DynamicCache issues
                "return_dict_in_generate": False,  # Simplified return format
            }
            
            # Add sampling parameters only if do_sample is True
            if self.do_sample:
                generation_kwargs.update({
                    "temperature": max(0.1, min(self.temperature, 2.0)),  # Clamp temperature
                    "top_p": max(0.1, min(self.top_p, 1.0)),  # Clamp top_p
                })
                
                # Only add repetition penalty if it's not too close to 1.0
                if abs(self.repetition_penalty - 1.0) > 0.05:
                    generation_kwargs["repetition_penalty"] = max(1.0, min(self.repetition_penalty, 2.0))
            
            with torch.no_grad():
                # Generate with explicit parameters
                outputs = self.model.generate(**generation_kwargs)
            
            # Extract only the new tokens
            if outputs.dim() > 1:
                generated_response_ids = outputs[0][encoded_chat.shape[1]:]
            else:
                generated_response_ids = outputs[encoded_chat.shape[1]:]
            
            # Decode the response
            generated_text = self.tokenizer.decode(
                generated_response_ids, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            ).strip()
            
            return generated_text
            
        except Exception as e:
            print(f"Manual generation error: {e}")
            # Ultra-simple fallback
            return self._simple_generate_fallback(messages)
    
    def _simple_generate_fallback(self, messages):
        """Ultra-simple generation as last resort."""
        try:
            # Get just the last user message for simple completion
            last_message = messages[-1]["content"] if messages else "Hello"
            
            # Simple tokenization
            inputs = self.tokenizer.encode(
                f"User: {last_message}\nAssistant:", 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=50,
                    do_sample=False,  # Greedy decoding for stability
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=False,
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs.shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            return response if response else "I'm having trouble generating a response right now."
            
        except Exception as e:
            print(f"Even simple fallback failed: {e}")
            return "I apologize, but I'm experiencing technical difficulties. Please try restarting the session."
    
    def generate_response(self, user_input):
        """Generate a response to user input."""
        # Add user's message to conversation
        self.current_messages.append({"role": "user", "content": user_input})
        
        print("Phi-3 is thinking...")
        start_time = time.time()
        
        try:
            # Use manual generation (pipeline disabled for Phi-3 compatibility)
            assistant_reply = self._generate_response_manual(self.current_messages)
            
            # Clean up the response
            assistant_reply = assistant_reply.strip()
            if not assistant_reply:
                assistant_reply = "I apologize, but I couldn't generate a proper response. Please try again."
            
            # Add assistant's response to conversation
            self.current_messages.append({"role": "assistant", "content": assistant_reply})
            
            # Save to history
            self.conversation_history.append({
                "user": user_input,
                "assistant": assistant_reply,
                "timestamp": datetime.now().isoformat()
            })
            
            end_time = time.time()
            print(f"\nPhi-3: {assistant_reply}")
            print(f"(Time taken: {end_time - start_time:.2f} seconds)")
            
        except Exception as e:
            print(f"Error generating response: {e}")
            print("Removing failed message from conversation...")
            
            # Remove the user message if generation completely failed
            if self.current_messages and self.current_messages[-1]["role"] == "user":
                self.current_messages.pop()
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.current_messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
        print("Conversation history reset.")
    
    def run_interactive_session(self):
        """Run the main interactive loop."""
        print(f"\n--- Interactive Phi-3 Session ---")
        print(f"Type 'help' for available commands or 'quit' to exit.")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit']:
                    if self.save_history and self.conversation_history:
                        self._save_conversation()
                    print("Exiting session. Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                elif user_input.lower() == 'reset':
                    self.reset_conversation()
                    continue
                
                elif user_input.lower() == 'save':
                    self._save_conversation()
                    continue
                
                elif user_input.lower() == 'show_history':
                    self._show_history()
                    continue
                
                elif user_input.lower().startswith('params '):
                    self._update_parameters(user_input.lower())
                    continue
                
                elif user_input.lower() == 'show_params':
                    self._show_parameters()
                    continue
                
                elif user_input.lower().startswith('system '):
                    system_message = user_input[7:].strip()
                    if system_message:
                        self._set_system_prompt(system_message)
                    else:
                        print("Please provide a system message. Example: system You are a creative writing assistant.")
                    continue
                
                # Generate response for regular input
                self.generate_response(user_input)
                
            except KeyboardInterrupt:
                print("\nCtrl+C detected. Exiting session. Goodbye!")
                if self.save_history and self.conversation_history:
                    self._save_conversation()
                break
            
            except Exception as e:
                print(f"An error occurred: {e}")
                print("Please try again or type 'reset' to clear the conversation.")

def main():
    parser = argparse.ArgumentParser(description="Interactive Phi-3 Chat Bot")
    parser.add_argument("--model", default=MODEL_NAME, help="Model name to use")
    parser.add_argument("--save-history", action="store_true", help="Save conversation history to file")
    
    args = parser.parse_args()
    
    try:
        bot = Phi3ChatBot(model_name=args.model, save_history=args.save_history)
        bot.run_interactive_session()
    except Exception as e:
        print(f"Failed to initialize chat bot: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
