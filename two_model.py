import torch
import time
import gc
import asyncio
import json
import os
import logging
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig # Keep BitsAndBytesConfig for reference or if you later want 4-bit

# --- 1. Configuration ---

# Model Configuration
SQL_MODEL_NAME = "defog/llama-3-sqlcoder-8b"  # For SQL generation
ANALYSIS_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct" # *** CHANGED TO PHI-3 MINI ***
API_ENDPOINT = "http://172.200.64.182:7860/execute"  # API for SQL execution

# File Path Configuration for saving results
SAVE_PATH = "/home/text_sql"
os.makedirs(SAVE_PATH, exist_ok=True)

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database Schema
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

# --- 2. FastAPI and Pydantic Setup ---

app = FastAPI(title="Dual-Model SQL Query Generator and Analyzer", version="2.1.0")
executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    success: bool
    question: str
    generated_sql: str
    sql_execution_result: dict
    llm_analysis: str
    sql_generation_time: float
    llm_analysis_time: float
    sql_execution_time: float
    total_processing_time: float
    file_saved: str

# --- 3. Optimized Device and Model Loading ---

def setup_device() -> torch.device:
    """Setup and configure GPU/CPU device with optimizations."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

        logger.info(f"✅ GPU detected: {gpu_name}")
        logger.info(f"📊 GPU Memory: {gpu_memory:.1f} GB")

        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        logger.warning("⚠️ GPU not available, using CPU.")
    return device

def load_sql_model_and_tokenizer(device: torch.device) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load SQL generation model and tokenizer with optimizations."""
    logger.info("🔄 Loading SQL model and tokenizer...")
    load_start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(SQL_MODEL_NAME, use_fast=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "low_cpu_mem_usage": True,
    }

    if torch.cuda.is_available():
        model_kwargs.update({
            "device_map": "auto",
            "load_in_8bit": True, # Keep 8-bit for Llama-3-SQLCoder
        })

    model = AutoModelForCausalLM.from_pretrained(SQL_MODEL_NAME, **model_kwargs)

    model.eval()
    if hasattr(model, 'config'):
        model.config.use_cache = True

    logger.info(f"🤖 SQL model loaded in {time.time() - load_start_time:.2f}s")
    return model, tokenizer

def load_analysis_model_and_tokenizer(device: torch.device) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load Phi-3 Mini analysis model and tokenizer with optimizations."""
    logger.info(f"🔄 Loading {ANALYSIS_MODEL_NAME} analysis model and tokenizer...")
    load_start_time = time.time()

    # For Phi-3-mini-4k-instruct, it's often best to load without explicit 8-bit
    # unless memory is a severe constraint, then consider load_in_4bit with BitsAndBytesConfig
    # For now, let's load it as is, or with float16 if GPU.
    tokenizer = AutoTokenizer.from_pretrained(ANALYSIS_MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None: # Phi-3 uses a specific chat template
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ '<|user|>\n' + message['content'] + '<|end|>\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '<|assistant|>\n' + message['content'] + '<|end|>\n' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|assistant|>' }}"
            "{% endif %}"
        )


    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32, # bfloat16 for Ampere+ GPUs, float16 for older, float32 for CPU
        "low_cpu_mem_usage": True,
    }

    if torch.cuda.is_available():
        model_kwargs.update({
            "device_map": "auto",
            # Optional: If you still face OOM with Phi-3 Mini, uncomment this:
            # "quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
            # "attn_implementation": "flash_attention_2", # Requires compatible GPU (Ampere or newer) and libraries
        })

    model = AutoModelForCausalLM.from_pretrained(ANALYSIS_MODEL_NAME, **model_kwargs)

    model.eval()
    if hasattr(model, 'config'):
        model.config.use_cache = True

    logger.info(f"🤖 {ANALYSIS_MODEL_NAME} analysis model loaded in {time.time() - load_start_time:.2f}s")
    return model, tokenizer

# --- 4. Prompt Engineering ---

def make_sql_prompt(question: str) -> str:
    """Creates a structured prompt for SQL generation."""
    return f"""### Task
Generate a T-SQL query that answers the following question. Use the provided schema.
Ensure the query is valid for Azure SQL Server.dont use NULLS LAST as its not part of T-SQL

### Schema
{DATABASE_SCHEMA}

### Question
{question}

### SQL
```sql
"""

def make_analysis_prompt(question: str, sql_query: str, sql_result: dict) -> str:
    """Creates a prompt for the analysis model (Phi-3 Mini) to analyze SQL results."""
    # Truncate long results to fit within context window
    result_str = json.dumps(sql_result, indent=2, default=str)
    if len(result_str) > 1500:
        result_str = result_str[:1500] + "\n... (results truncated for brevity)"

    # Extract relevant data from sql_result for clearer prompting
    actual_result_value = None
    if sql_result and sql_result.get('success') and sql_result.get('data') and len(sql_result['data']) > 0:
        first_row = sql_result['data'][0]
        if first_row:
            # Assuming the value is the first one in the dict for aggregate results
            actual_result_value = next(iter(first_row.values()), None)

    # Use Phi-3's instruction format for clarity
    messages = [
        {"role": "user", "content": f"""
You are a data analyst. Your task is to provide a concise, direct, natural language answer to the original question, based ONLY on the provided SQL query results.

Do NOT generate any code (Python, SQL, R, JSON, etc.), functions, or external commands.
Do NOT include any introductory phrases like "Based on the results," "The analysis shows," or "According to the data."
Do NOT include any concluding remarks or conversational filler.
Just provide the direct answer or a summary.

Original Question: {question}

Executed SQL Query:
{sql_query}

SQL Execution Result (Raw JSON):
{result_str}

"""}
    ]

    # Add specific guidance if a clear single value result is found
    if actual_result_value is not None:
        messages[0]["content"] += f"\n\nSpecifically, the numerical result for this query is: {actual_result_value}\n\nGiven the question '{question}', what is the direct natural language answer based ONLY on the result '{actual_result_value}'?"
    else:
        messages[0]["content"] += f"\n\nGiven the question '{question}', provide a direct natural language summary of the above SQL Execution Result."

    # Apply the tokenizer's chat template
    # Phi-3 models use the tokenizer.apply_chat_template for correct formatting
    # ensure add_generation_prompt=True to tell the model it's its turn to generate
    return app.state.analysis_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# --- 5. High-Performance Inference ---

@contextmanager
def inference_mode():
    """Context manager for optimized inference."""
    with torch.no_grad():
        if torch.cuda.is_available():
            # Use bfloat16 if GPU supports it, otherwise float16 or float32
            # Phi-3 typically runs well with bfloat16
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                yield
        else:
            yield

def generate_text_optimized(prompt: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, max_new_tokens: int, temperature: float = 0.0) -> str:
    """Highly optimized text generation function."""
    # Ensure correct padding_side for generation
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left" # Typically for generation to pad inputs

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)

    # Move inputs to the same device as the model
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device, non_blocking=True) for k, v in inputs.items()}

    input_length = inputs['input_ids'].shape[1]

    with inference_mode():
        gen_kwargs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs.get('attention_mask'),
            'max_new_tokens': max_new_tokens,
            'do_sample': temperature > 0,
            'num_beams': 1 if temperature == 0 else 1, # Keep num_beams=1 for analysis for directness
            'pad_token_id': tokenizer.eos_token_id,
            'use_cache': True,
            'early_stopping': True,
        }
        if temperature > 0:
            gen_kwargs['temperature'] = temperature
            gen_kwargs['top_p'] = 0.9 # Good default for sampling

        outputs = model.generate(**gen_kwargs)

    # Reset padding side
    tokenizer.padding_side = original_padding_side

    # Decode and post-process
    # For instruct models, output often includes the prompt plus the assistant's turn
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # For instruct models, specifically extract the assistant's part
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        # This will depend on the exact chat template. For Phi-3, it's usually everything after '<|assistant|>'
        assistant_tag = '<|assistant|>'
        if assistant_tag in generated_text:
            generated_text = generated_text.split(assistant_tag, 1)[1].strip()
        # Further clean up any potential trailing end tokens if they somehow made it through skip_special_tokens
        generated_text = generated_text.replace('<|end|>', '').strip()


    # Aggressive cleanup
    del inputs, outputs, gen_kwargs
    cleanup_memory()

    return generated_text

def extract_sql_from_response(response: str) -> str:
    """Extracts SQL code from the model's response."""
    # This logic remains the same for the SQLCoder model
    if "```sql" in response:
        start = response.find("```sql") + 6
        end = response.find("```", start)
        if end != -1:
            return response[start:end].strip()
    elif "sql" in response.lower(): # Fallback for less structured output
        start = response.lower().find("sql") + 3
        lines = response[start:].split('\n')
        sql_lines = []
        for line in lines:
            if line.strip() and not line.strip().startswith('#'):
                sql_lines.append(line)
            elif sql_lines: # Stop if we encounter an empty line after starting SQL
                break
        return '\n'.join(sql_lines).strip()
    return response.strip()

# --- 6. Core Logic Functions ---

def generate_sql(question: str) -> Tuple[str, float]:
    """Generates SQL query from a question using SQLCoder."""
    start_time = time.time()
    prompt = make_sql_prompt(question)
    response = generate_text_optimized(prompt, app.state.sql_model, app.state.sql_tokenizer, max_new_tokens=300)
    sql_query = extract_sql_from_response(response)
    generation_time = time.time() - start_time
    logger.info(f"SQL generated in {generation_time:.2f}s")
    return sql_query, generation_time

def analyze_results(question: str, sql_query: str, sql_result: dict) -> Tuple[str, float]:
    """Generates a natural language analysis using Phi-3 Mini."""
    start_time = time.time()
    prompt = make_analysis_prompt(question, sql_query, sql_result)
    analysis = generate_text_optimized(
        prompt,
        app.state.analysis_model,
        app.state.analysis_tokenizer,
        max_new_tokens=100,  # Max tokens for a concise analysis
        temperature=0.1,     # Low temperature for direct, less creative output
        num_beams=1          # Use greedy decoding for directness
    )

    # Post-processing to ensure no code or filler phrases
    analysis = analysis.replace("```python", "").replace("```sql", "").replace("```", "").strip()
    analysis_lines = [line for line in analysis.split('\n') if not (
        line.strip().startswith('import') or
        line.strip().startswith('from') or
        line.strip().startswith('df = pd.read_sql_query') or
        line.strip().startswith('print(') or
        line.strip() == '"""' # Remove triple quotes if model outputs them
    )]
    analysis = "\n".join(analysis_lines).strip()

    analysis_time = time.time() - start_time
    logger.info(f"Analysis generated in {analysis_time:.2f}s using {ANALYSIS_MODEL_NAME}")
    return analysis, analysis_time

def execute_sql(sql_query: str) -> Tuple[Dict[str, Any], float]:
    """Executes SQL query via the external API."""
    start_time = time.time()
    try:
        response = requests.post(
            API_ENDPOINT,
            json={"query": sql_query},
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as e:
        logger.error(f"SQL execution API error: {e}")
        # Capture more specific error details if available
        error_details = {"error": str(e)}
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_details.update(e.response.json())
                error_details["status_code"] = e.response.status_code
            except json.JSONDecodeError:
                error_details["raw_response"] = e.response.text
                error_details["status_code"] = e.response.status_code
        result = error_details

    execution_time = time.time() - start_time
    logger.info(f"SQL executed in {execution_time:.2f}s")
    return result, execution_time

def save_result_to_file(data: dict) -> str:
    """Saves the full transaction to a timestamped file."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(SAVE_PATH, f"query_{timestamp}.json")
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4, default=str)
        logger.info(f"Result saved to {filename}")
        return filename
    except IOError as e:
        logger.error(f"Failed to save file: {e}")
        return f"Error saving file: {e}"

def cleanup_memory():
    """Aggressive memory cleanup."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

# --- 7. FastAPI Application Events and Endpoints ---

@app.on_event("startup")
async def startup_event():
    """Load both models on application startup."""
    app.state.device = setup_device()

    # Load SQL generation model
    app.state.sql_model, app.state.sql_tokenizer = load_sql_model_and_tokenizer(app.state.device)

    # Load analysis model
    app.state.analysis_model, app.state.analysis_tokenizer = load_analysis_model_and_tokenizer(app.state.device)

    app.state.loop = asyncio.get_event_loop()
    logger.info("🚀 Both models loaded successfully!")

@app.on_event("shutdown")
def shutdown_event():
    """Clean up resources on shutdown."""
    cleanup_memory()
    executor.shutdown(wait=True)
    logger.info("🧹 Memory cleaned up and executor shut down.")

@app.post("/generate-and-analyze-sql", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Main endpoint to process a natural language question through the full
    generate -> execute -> analyze pipeline using dual models.
    """
    total_start_time = time.time()
    loop = app.state.loop

    sql_query = ""
    sql_generation_time = 0.0
    sql_execution_result = {"error": "SQL execution not attempted."}
    sql_execution_time = 0.0
    llm_analysis = "Analysis not performed."
    llm_analysis_time = 0.0
    overall_success = False

    try:
        # 1. Generate SQL using SQLCoder
        sql_query, sql_generation_time = await loop.run_in_executor(
            executor, generate_sql, request.question
        )

        if not sql_query:
            logger.error("Failed to generate SQL query.")
            llm_analysis = "Failed to generate SQL query."
            raise HTTPException(status_code=400, detail="Failed to generate SQL query.")

        # 2. Execute SQL
        sql_execution_result, sql_execution_time = await loop.run_in_executor(
            executor, execute_sql, sql_query
        )

        # Check if SQL execution itself was successful
        if "error" in sql_execution_result and sql_execution_result["error"] is not None:
            logger.error(f"SQL execution failed: {sql_execution_result['error']}")
            llm_analysis = f"SQL execution failed with error: {sql_execution_result['error']}"
            overall_success = False
        else:
            # 3. Analyze Results using Phi-3 Mini
            llm_analysis, llm_analysis_time = await loop.run_in_executor(
                executor, analyze_results, request.question, sql_query, sql_execution_result
            )

            # Heuristic check for analysis quality: look for common code patterns or empty analysis
            # Refine these patterns based on what the model might incorrectly output
            if any(p in llm_analysis.lower() for p in ['import ', 'def ', 'class ', 'select ', 'from ', 'join ', '```', 'json.dumps', 'pd.read_sql_query']):
                logger.warning("LLM analysis still contains suspected code patterns. Marking overall success as false.")
                llm_analysis = "Analysis failed to provide natural language insights (generated code or unexpected patterns)."
                overall_success = False
            elif not llm_analysis.strip():
                logger.warning("LLM analysis is empty after generation. Marking overall success as false.")
                llm_analysis = "LLM analysis provided an empty response."
                overall_success = False
            else:
                overall_success = True # All steps succeeded and analysis looks good

    except HTTPException as e:
        # Re-raise explicit HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        llm_analysis = f"An internal server error occurred during processing: {str(e)}"
        overall_success = False # Catch-all for other errors

    total_processing_time = time.time() - total_start_time
    response_data = {
        "success": overall_success,
        "question": request.question,
        "generated_sql": sql_query,
        "sql_execution_result": sql_execution_result,
        "llm_analysis": llm_analysis,
        "sql_generation_time": round(sql_generation_time, 2),
        "llm_analysis_time": round(llm_analysis_time, 2),
        "sql_execution_time": round(sql_execution_time, 2),
        "total_processing_time": round(total_processing_time, 2),
    }

    # 4. Save results - always try to save even on failure
    file_saved = await loop.run_in_executor(
        executor, save_result_to_file, response_data
    )

    response_data["file_saved"] = file_saved
    return QueryResponse(**response_data)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "sql_model_loaded": hasattr(app.state, 'sql_model'),
        "analysis_model_loaded": hasattr(app.state, 'analysis_model'),
        "device": str(app.state.device) if hasattr(app.state, 'device') else "unknown"
    }

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint."""
    return {"message": "Dual-Model SQL Generation and Analysis API is running with Phi-3 Mini for analysis."}

# --- 8. Additional Utility Endpoints ---

@app.post("/generate-sql-only")
async def generate_sql_only(request: QueryRequest):
    """Generate SQL query without execution or analysis."""
    try:
        sql_query, generation_time = await app.state.loop.run_in_executor(
            executor, generate_sql, request.question
        )
        return {
            "question": request.question,
            "generated_sql": sql_query,
            "generation_time": round(generation_time, 2)
        }
    except Exception as e:
        logger.error(f"SQL generation error: {e}")
        raise HTTPException(status_code=500, detail=f"SQL generation failed: {str(e)}")

@app.get("/memory-status")
async def memory_status():
    """Get current memory usage statistics."""
    memory_info = {}

    if torch.cuda.is_available():
        memory_info["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1e9
        memory_info["gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1e9
        memory_info["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1e9

    import psutil
    process = psutil.Process()
    memory_info["cpu_memory_mb"] = process.memory_info().rss / 1024 / 1024

    return memory_info

@app.get("/model-info")
async def model_info():
    """Get information about loaded models."""
    return {
        "sql_model": SQL_MODEL_NAME,
        "analysis_model": ANALYSIS_MODEL_NAME,
        "sql_model_loaded": hasattr(app.state, 'sql_model'),
        "analysis_model_loaded": hasattr(app.state, 'analysis_model'),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
