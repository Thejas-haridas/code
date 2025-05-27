import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import time

# --- Configuration ---
# Choose the Phi-3 model you want to test.
# Examples: "microsoft/Phi-3-small-8k-instruct", "microsoft/Phi-3-mini-4k-instruct"
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct" # A good starting point for local testing

# Generation parameters (can be adjusted)
DEFAULT_MAX_NEW_TOKENS = 200
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Load Model and Tokenizer ---
print(f"\nLoading model: {MODEL_NAME}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32, # Use float16 for GPU for faster inference
        device_map="auto" # Automatically map model layers to available devices (GPU/CPU)
    )
    # Ensure the model is on the correct device
    model.to(device)
    model.eval() # Set model to evaluation mode
    print("Model loaded successfully!")
    print(f"Type 'quit' or 'exit' to stop the interactive session.")
    print(f"Type 'help' for options.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please check the model name, your internet connection, or accept terms on Hugging Face.")
    exit()

# Set up the pipeline for easier interaction with instruct models
# The pipeline handles the chat template automatically if the tokenizer has one.
try:
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        trust_remote_code=True
    )
except Exception as e:
    print(f"Error initializing pipeline: {e}")
    print("Proceeding with manual tokenization. Some instruct model features might be less optimized.")
    pipe = None # Fallback to manual if pipeline fails

# --- Interactive Loop ---
print("\n--- Interactive Phi-3 Session ---")
current_messages = [{"role": "system", "content": "You are a helpful AI assistant."}]

while True:
    try:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ['quit', 'exit']:
            print("Exiting session. Goodbye!")
            break
        elif user_input.lower() == 'help':
            print("\n--- Help ---")
            print("Type your prompt and press Enter.")
            print(f"  - 'quit' or 'exit': Stop the session.")
            print(f"  - 'reset': Clear the conversation history.")
            print(f"  - 'params max_new_tokens=<num> temp=<val> top_p=<val>': Change generation parameters.")
            print(f"    Example: params max_new_tokens=100 temp=0.5 top_p=0.8")
            print(f"  - 'show_params': Display current generation parameters.")
            print("----------------")
            continue
        elif user_input.lower() == 'reset':
            current_messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
            print("Conversation history reset.")
            continue
        elif user_input.lower().startswith('params '):
            parts = user_input.split(' ')
            for part in parts[1:]:
                if part.startswith('max_new_tokens='):
                    try:
                        DEFAULT_MAX_NEW_TOKENS = int(part.split('=')[1])
                        print(f"Set max_new_tokens to {DEFAULT_MAX_NEW_TOKENS}")
                    except ValueError:
                        print("Invalid value for max_new_tokens. Must be an integer.")
                elif part.startswith('temp='):
                    try:
                        DEFAULT_TEMPERATURE = float(part.split('=')[1])
                        print(f"Set temperature to {DEFAULT_TEMPERATURE}")
                    except ValueError:
                        print("Invalid value for temp. Must be a float.")
                elif part.startswith('top_p='):
                    try:
                        DEFAULT_TOP_P = float(part.split('=')[1])
                        print(f"Set top_p to {DEFAULT_TOP_P}")
                    except ValueError:
                        print("Invalid value for top_p. Must be a float.")
            continue
        elif user_input.lower() == 'show_params':
            print(f"\nCurrent Generation Parameters:")
            print(f"  max_new_tokens: {DEFAULT_MAX_NEW_TOKENS}")
            print(f"  temperature: {DEFAULT_TEMPERATURE}")
            print(f"  top_p: {DEFAULT_TOP_P}")
            continue

        # Add user's message to the conversation history
        current_messages.append({"role": "user", "content": user_input})

        print("Phi-3 is thinking...")
        start_time = time.time()

        if pipe:
            # Use the pipeline for instruction following
            response = pipe(
                current_messages,
                max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=DEFAULT_TEMPERATURE,
                top_p=DEFAULT_TOP_P,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id # Important for batching, though less critical for single interactive input
            )
            generated_text_full = response[0]['generated_text']

            # Extract only the assistant's last reply
            # The chat template adds specific markers. We need to find the last assistant turn.
            # Example: <|system|>...<|end|><|user|>...<|end|><|assistant|>...<|end|>
            # We want the content after the final <|assistant|> and before its <|end|>
            assistant_marker = "<|assistant|>"
            end_marker = "<|end|>"

            # Find the last assistant marker
            last_assistant_index = generated_text_full.rfind(assistant_marker)
            if last_assistant_index != -1:
                # Get the part of the string starting from the last assistant marker
                assistant_response_part = generated_text_full[last_assistant_index + len(assistant_marker):]
                # Find the first end_marker after the assistant's response
                end_index = assistant_response_part.find(end_marker)
                if end_index != -1:
                    assistant_reply = assistant_response_part[:end_index].strip()
                else:
                    assistant_reply = assistant_response_part.strip() # No end marker, take till end
            else:
                # Fallback if the assistant marker isn't found (e.g., first turn, or non-chat-templated model)
                # In this case, just show the full generated text, which might include the prompt
                assistant_reply = generated_text_full.strip()

            print(f"\nPhi-3: {assistant_reply}")
            # Add assistant's reply to the conversation history for multi-turn chats
            current_messages.append({"role": "assistant", "content": assistant_reply})

        else: # Manual tokenization fallback if pipeline failed
            # Apply chat template manually
            # This is crucial for instruct models to understand multi-turn conversation context
            # Phi-3 instruct models use a specific format
            # e.g., <|system|>You are a helpful AI assistant.<|end|><|user|>What is X?<|end|>
            # Note: `apply_chat_template` is the recommended way to format inputs for instruct models.
            encoded_chat = tokenizer.apply_chat_template(
                current_messages,
                tokenize=True,
                add_generation_prompt=True, # Add the special token that indicates the model should generate
                return_tensors="pt"
            ).to(device)

            outputs = model.generate(
                encoded_chat,
                max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                num_return_sequences=1,
                temperature=DEFAULT_TEMPERATURE,
                top_p=DEFAULT_TOP_P,
                do_sample=True, # Enable sampling for temperature/top_p
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id # Stop generation at EOS
            )

            # Decode the generated text. Note that `outputs` will contain the input prompt as well.
            # We need to slice it to get only the new tokens generated by the model.
            generated_response_ids = outputs[0][encoded_chat.shape[1]:] # Get only the new tokens
            generated_text = tokenizer.decode(generated_response_ids, skip_special_tokens=True).strip()
            print(f"\nPhi-3: {generated_text}")
            # Add assistant's reply to the conversation history
            current_messages.append({"role": "assistant", "content": generated_text})


        end_time = time.time()
        print(f"(Time taken: {end_time - start_time:.2f} seconds)")

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exiting session. Goodbye!")
        break
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please try again or 'reset' the conversation.")
        # Optionally, you might want to remove the last user message if it caused an error
        if current_messages and current_messages[-1]["role"] == "user":
            current_messages.pop() # Remove the problematic user input
