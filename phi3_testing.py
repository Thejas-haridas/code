import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import time

# --- Configuration ---
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

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
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" # Let accelerate manage device placement
    )
    model.eval() # Set model to evaluation mode
    print("Model loaded successfully!")
    print(f"Type 'quit' or 'exit' to stop the interactive session.")
    print(f"Type 'help' for options.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please check the model name, your internet connection, or accept terms on Hugging Face.")
    exit()

# Set up the pipeline for easier interaction with instruct models
try:
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=True
    )
except Exception as e:
    print(f"Error initializing pipeline: {e}")
    print("Proceeding with manual tokenization. Some instruct model features might be less optimized.")
    pipe = None

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
            response = pipe(
                current_messages,
                max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=DEFAULT_TEMPERATURE,
                top_p=DEFAULT_TOP_P,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated_text_full = response[0]['generated_text']

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
                assistant_reply = generated_text_full.strip()

            print(f"\nPhi-3: {assistant_reply}")
            current_messages.append({"role": "assistant", "content": assistant_reply})

        else: # Manual tokenization fallback if pipeline failed
            encoded_chat = tokenizer.apply_chat_template(
                current_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)

            outputs = model.generate(
                encoded_chat,
                max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                num_return_sequences=1,
                temperature=DEFAULT_TEMPERATURE,
                top_p=DEFAULT_TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                attention_mask=encoded_chat.ne(tokenizer.pad_token_id).long()
            )

            generated_response_ids = outputs[0][encoded_chat.shape[1]:]
            generated_text = tokenizer.decode(generated_response_ids, skip_special_tokens=True).strip()
            print(f"\nPhi-3: {generated_text}")
            current_messages.append({"role": "assistant", "content": generated_text})

        end_time = time.time()
        print(f"(Time taken: {end_time - start_time:.2f} seconds)")

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exiting session. Goodbye!")
        break
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please try again or 'reset' the conversation.")
        if current_messages and current_messages[-1]["role"] == "user":
            current_messages.pop()
