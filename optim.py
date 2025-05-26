import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
from contextlib import contextmanager

# --- 1. Configuration ---

MODEL_NAME = "defog/sqlcoder-7b-2"

DATABASE_SCHEMA = """
create table dwh.dim_claims (claim_reference_id varchar(MAX), date_claim_first_notified
datetime, date_of_loss_from datetime, date_claim_opened datetime, date_of_loss_to
datetime, cause_of_loss_code varchar(MAX), loss_description varchar(MAX),
date_coverage_confirmed datetime, date_closed datetime, date_claim_amount_agreed
datetime, date_paid_final_amount datetime, date_fees_paid_final_amount datetime,
date_reopened datetime, date_claim_denied datetime, date_claim_withdrawn datetime,
status varchar(MAX), refer_to_underwriters varchar(MAX), denial_indicator varchar(MAX),
reason_for_denial varchar(MAX), claim_total_claimed_amount decimal,
settlement_currency_code varchar(MAX), indemnity_amount_paid decimal,
fees_amount_paid decimal, expenses_paid_amount decimal, dw_ins_upd_dt datetime,
org_id varchar(MAX))

create table dwh.dim_policy (Id bigint, agreement_id varchar(MAX), policy_number
varchar(MAX), new_or_renewal varchar(MAX), group_reference varchar(MAX),
broker_reference varchar(MAX), changed_date datetime, effective_date datetime,
start_date_time datetime, expiry_date_time datetime, renewal_date_time datetime,
product_code varchar(MAX), product_name varchar(MAX), country_code varchar(MAX),
country varchar(MAX), a3_country_code varchar(MAX), country_sub_division_code
varchar(MAX), class_of_business_code varchar(MAX), classof_business_name
varchar(MAX), main_line_of_business_name varchar(MAX), insurance_type varchar(MAX),
section_details_number varchar(MAX), section_details_code varchar(MAX),
section_details_name varchar(MAX), line_of_business varchar(MAX),
section_details_description varchar(MAX), dw_ins_upd_dt datetime, org_id varchar(MAX),
document_id varchar(MAX))

create table dwh.fact_claims_dtl (Id bigint, claim_reference_id varchar(MAX), agreement_id
varchar(MAX), policy_number varchar(MAX), org_id varchar(MAX), riskitems_id
varchar(MAX), Payment_Detail_Settlement_Currency_Code varchar(MAX), Paid_Amount
decimal, Expenses_Paid_Total_Amount decimal, Coverage_Legal_Fees_Total_Paid_Amount
decimal, Defence_Legal_Fees_Total_Paid_Amount decimal,
Adjusters_Fees_Total_Paid_Amount decimal, TPAFees_Paid_Amount decimal,
Fees_Paid_Amount decimal, Incurred_Detail_Settlement_Currency_Code varchar(MAX),
Indemnity_Amount decimal, Expenses_Amount decimal, Coverage_Legal_Fees_Amount
decimal, Defence_Fees_Amount decimal, Adjuster_Fees_Amount decimal,
TPAFees_Amount decimal, Fees_Amount decimal, indemnity_reserves_amount decimal,
dw_ins_upd_dt datetime, indemnity_amount_paid decimal)

create table dwh.fact_premium (Id bigint, agreement_id varchar(MAX), policy_number
varchar(MAX), org_id varchar(MAX), riskitems_id varchar(MAX), original_currency_code
varchar(MAX), total_paid decimal, instalments_amount decimal, taxes_amount_paid
decimal, commission_percentage decimal, commission_amount_paid decimal,
brokerage_amount_paid decimal, insurance_amount_paid decimal, additional_fees_paid
decimal, settlement_currency_code varchar(MAX), gross_premium_settlement_currency
decimal, brokerage_amount_paid_settlement_currency decimal,
net_premium_settlement_currency decimal,
commission_amount_paid_settlement_currency decimal,
final_net_premium_settlement_currency decimal, rate_of_exchange decimal,
total_settlement_amount_paid decimal, date_paid datetime, transaction_type
varchar(MAX), net_amount decimal, gross_premium_paid_this_time decimal,
final_net_premium decimal, tax_amount decimal, dw_ins_upd_dt datetime)

create table dwh.fct_policy (Id bigint, agreement_id varchar(MAX), policy_number
varchar(MAX), org_id varchar(MAX), start_date date, annual_premium decimal,
sum_insured decimal, limit_of_liability decimal, final_net_premium decimal, tax_amount
decimal, final_net_premium_settlement_currency varchar(MAX),
settlement_currency_code varchar(MAX), gross_premium_before_taxes_amount decimal,
dw_ins_upd_dt datetime, document_id varchar(MAX), gross_premium_paid_this_time
decimal)
"""

JOIN_RULES = """
Based on the user's question, generate an SQL query. The query should always include
use the logic and give accurate results 
necessary table joins based on the following rules:
- For a policy-related question, join 'fct_policy' and 'dim_policy' on 'policy_number' and 'org_id'.
- For a claims-related question, join 'fact_claims_dtl' and 'dim_claims' on 'claim_reference_id' and 'org_id'.
- Always use proper T-SQL syntax.
- Return only the SQL query without any explanatory text.
"""

# --- 2. GPU and Device Configuration with Optimizations ---

def setup_device():
    """Setup and configure GPU/CPU device with optimizations"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_count = torch.cuda.device_count()
        current_gpu = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_gpu)
        gpu_memory = torch.cuda.get_device_properties(current_gpu).total_memory / 1e9
        
        print(f"✅ GPU detected: {gpu_name}")
        print(f"📊 GPU Memory: {gpu_memory:.1f} GB")
        print(f"🔢 Available GPUs: {gpu_count}")
        print(f"⚡ Using CUDA device: {current_gpu}")
        
        # Clear GPU cache and optimize settings
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster computation
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # Optimize cudnn for consistent input sizes
        
    else:
        device = torch.device("cpu")
        print("⚠️  GPU not available, using CPU")
        # CPU optimizations
        torch.set_num_threads(torch.get_num_threads())
        
    return device

# --- 3. Optimized Prompt Construction with Caching ---

class PromptCache:
    """Cache for reusable prompt components"""
    def __init__(self):
        self.base_prompt_template = None
        self._setup_template()
    
    def _setup_template(self):
        """Pre-compile the base template"""
        self.base_prompt_template = f"""### Task
### Output Requirements

Use valid T-SQL syntax for Azure SQL Server. Avoid using PostgreSQL-specific features (like "::FLOAT"). Use CAST(... AS FLOAT) for casting.

Generate a SQL query to answer [QUESTION]{{}}[/QUESTION]

### Database Schema

The query will run on a database with the following schema:

{DATABASE_SCHEMA}

### Join Rules

{JOIN_RULES}

### Answer

Given the database schema and join rules, here is the SQL query that answers [QUESTION]{{}}[/QUESTION]

[SQL]
"""
    
    def make_prompt(self, user_question):
        """Fast prompt construction using pre-compiled template"""
        return self.base_prompt_template.format(user_question, user_question)

# Global prompt cache
prompt_cache = PromptCache()

# --- 4. Optimized Model Loading with Better Memory Management ---

def load_model_and_tokenizer():
    """Load model and tokenizer with advanced optimizations"""
    device = setup_device()
    
    print("\n🔄 Loading model and tokenizer...")
    load_start_time = time.time()
    
    # Load tokenizer with optimizations
    tokenizer_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,  # Use fast tokenizer if available
        padding_side="left"  # Better for generation
    )
    
    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer_time = time.time() - tokenizer_start
    print(f"📝 Tokenizer loaded in {tokenizer_time:.2f}s")
    
    # Advanced model loading parameters
    model_kwargs = {
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    
    if torch.cuda.is_available():
        # GPU-specific optimizations
        model_kwargs.update({
            "device_map": "auto",
            "load_in_8bit": True,  # 8-bit quantization for memory efficiency
            "llm_int8_enable_fp32_cpu_offload": True,  # Offload some layers to CPU if needed
        })
    
    # Load model with error handling
    model_start = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
    except Exception as e:
        print(f"⚠️  Failed to load with 8-bit, trying 16-bit: {e}")
        model_kwargs.pop('load_in_8bit', None)
        model_kwargs.pop('llm_int8_enable_fp32_cpu_offload', None)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
    
    model_time = time.time() - model_start
    
    # Model optimizations
    if not torch.cuda.is_available():
        model = model.to(device)
    
    # Enable optimizations
    model.eval()  # Set to evaluation mode
    if hasattr(model, 'config'):
        model.config.use_cache = True  # Enable KV caching
    
    total_load_time = time.time() - load_start_time
    
    print(f"🤖 Model loaded in {model_time:.2f}s")
    print(f"⏱️  Total loading time: {total_load_time:.2f}s")
    
    # Memory usage reporting
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"💾 GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
    
    return model, tokenizer, device

# --- 5. High-Performance Inference with Multiple Optimizations ---

@contextmanager
def inference_mode():
    """Context manager for optimized inference"""
    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.amp.autocast('cuda'):
                yield
        else:
            with torch.amp.autocast('cpu'):
                yield

def generate_sql_optimized(user_question, model, tokenizer, device):
    """Highly optimized SQL generation with minimal overhead"""
    print(f"\n🤔 Processing question: '{user_question}'")
    
    total_start = time.time()
    
    # Fast prompt construction using cache
    prompt_start = time.time()
    prompt = prompt_cache.make_prompt(user_question)
    prompt_time = time.time() - prompt_start
    
    # Efficient tokenization
    tokenize_start = time.time()
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",
        truncation=True,
        max_length=2048,  # Reasonable limit to prevent excessive context
        padding=False  # No padding needed for single input
    )
    
    # Move to device efficiently
    if torch.cuda.is_available():
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
    
    tokenize_time = time.time() - tokenize_start
    input_length = inputs['input_ids'].shape[1]
    print(f"📏 Input tokens: {input_length}")
    
    # Optimized generation
    generate_start = time.time()
    
    with inference_mode():
        # Prepare generation arguments, avoiding duplicate attention_mask
        gen_kwargs = {
            'input_ids': inputs['input_ids'],
            'max_new_tokens': 200,
            'do_sample': False,
            'num_beams': 1,
            'pad_token_id': tokenizer.eos_token_id,
            'use_cache': True,
            'early_stopping': True,
        }
        
        # Add attention_mask only if it exists and is not already in inputs
        if 'attention_mask' in inputs:
            gen_kwargs['attention_mask'] = inputs['attention_mask']
        
        outputs = model.generate(**gen_kwargs)
    
    generate_time = time.time() - generate_start
    
    # Fast decoding and extraction
    decode_start = time.time()
    # Only decode the new tokens
    new_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Fast SQL extraction
    sql_query = generated_text.strip()
    if "[SQL]" in sql_query:
        sql_query = sql_query.split("[SQL]")[-1].strip()
    
    decode_time = time.time() - decode_start
    total_time = time.time() - total_start
    
    # Performance metrics
    output_tokens = len(new_tokens)
    tokens_per_second = output_tokens / generate_time if generate_time > 0 else 0
    
    # Streamlined timing report
    print(f"\n⏱️  Performance: {total_time:.2f}s total ({tokens_per_second:.1f} tok/s)")
    
    return sql_query

# --- 6. Batch Processing for Multiple Queries ---

def generate_sql_batch(questions, model, tokenizer, device, batch_size=4):
    """Process multiple questions in batches for better throughput"""
    results = []
    
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        batch_start = time.time()
        
        # Prepare batch prompts
        prompts = [prompt_cache.make_prompt(q) for q in batch]
        
        # Tokenize batch
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        
        # Generate batch
        with inference_mode():
            # Prepare generation arguments for batch
            gen_kwargs = {
                'input_ids': inputs['input_ids'],
                'max_new_tokens': 200,
                'do_sample': False,
                'num_beams': 1,
                'pad_token_id': tokenizer.eos_token_id,
                'use_cache': True,
                'early_stopping': True,
            }
            
            # Add attention_mask if present
            if 'attention_mask' in inputs:
                gen_kwargs['attention_mask'] = inputs['attention_mask']
            
            outputs = model.generate(**gen_kwargs)
        
        # Decode batch results
        input_length = inputs['input_ids'].shape[1]
        for j, output in enumerate(outputs):
            new_tokens = output[input_length:]
            generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            sql_query = generated_text.strip()
            if "[SQL]" in sql_query:
                sql_query = sql_query.split("[SQL]")[-1].strip()
            results.append(sql_query)
        
        batch_time = time.time() - batch_start
        print(f"📦 Batch {i//batch_size + 1}: {len(batch)} queries in {batch_time:.2f}s")
    
    return results

# --- 7. Memory Management Utilities ---

def cleanup_memory():
    """Aggressive memory cleanup"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

# --- 8. Main Execution with Optimizations ---

def main():
    """Optimized main execution function"""
    try:
        # Load model and tokenizer
        model, tokenizer, device = load_model_and_tokenizer()
        
        print("\n" + "="*60)
        print("🚀 Optimized SQL Generator Ready!")
        print("💡 Performance optimizations enabled")
        print("="*60)
        
        # Interactive loop
        while True:
            try:
                user_input = input("\n❓ Enter your question, 'batch' for batch demo, or 'quit' to exit: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'batch':
                    # Batch processing demo
                    demo_questions = [
                        "Show me total claims by status",
                        "What is the average premium by country?",
                        "List all policies expiring next month",
                        "Show claims with highest paid amounts"
                    ]
                    print(f"\n📦 Processing {len(demo_questions)} questions in batch...")
                    batch_results = generate_sql_batch(demo_questions, model, tokenizer, device)
                    
                    for q, sql in zip(demo_questions, batch_results):
                        print(f"\n❓ {q}")
                        print(f"🎯 {sql}")
                    continue
                
                if not user_input:
                    print("⚠️  Please enter a question.")
                    continue
                
                # Generate SQL with optimized function
                sql_query = generate_sql_optimized(user_input, model, tokenizer, device)
                
                print(f"\n🎯 Generated SQL Query:")
                print("-" * 40)
                print(sql_query)
                print("-" * 40)
                
                # Optional memory cleanup every few queries
                cleanup_memory()
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error during inference: {str(e)}")
                continue
                
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return 1
    
    finally:
        # Final cleanup
        cleanup_memory()
        print("🧹 Memory cleaned up")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
