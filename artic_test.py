import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "Snowflake/Arctic-Text2SQL-R1-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model with GPU optimization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use half precision for memory efficiency
    device_map="auto",          # Automatically distribute across available GPUs
    trust_remote_code=True      # Required for some models
)

prompt = (
    "Given the table 'employees' with columns 'id', 'name', 'department', 'salary', "
    "write an SQL query to find the names of employees in the 'Engineering' department earning more than $100,000."
)

# Tokenize and move to device
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate with GPU acceleration
with torch.no_grad():  # Disable gradient computation for inference
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,        # Deterministic generation
        temperature=0.1,        # Low temperature for more focused output
        pad_token_id=tokenizer.eos_token_id
    )

# Decode only the new tokens (excluding the input prompt)
sql_query = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print("Generated SQL Query:")
print(sql_query)
