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
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- 1. Configuration ---

# Model Configuration
SQL_MODEL_NAME = "defog/llama-3-sqlcoder-8b"  # For SQL generation
ANALYSIS_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"  # For analysis
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
            "load_in_8bit": True,
        })

    model = AutoModelForCausalLM.from_pretrained(SQL_MODEL_NAME, **model_kwargs)
    
    model.eval()
    if hasattr(model, 'config'):
        model.config.use_cache = True

    logger.info(f"🤖 SQL model loaded in {time.time() - load_start_time:.2f}s")
    return model, tokenizer

def load_analysis_model_and_tokenizer(device: torch.device) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load Phi-3-mini analysis model and tokenizer with optimizations."""
    logger.info("🔄 Loading Phi-3-mini analysis model and tokenizer...")
    load_start_time = time.time()

    # Setup quantization for Phi-3-mini
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    tokenizer = AutoTokenizer.from_pretrained(ANALYSIS_MODEL_NAME, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "quantization_config": quantization_config,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }

    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(ANALYSIS_MODEL_NAME, **model_kwargs)
    
    model.eval()
    if hasattr(model, 'config'):
        model.config.use_cache = True

    logger.info(f"🤖 Phi-3-mini analysis model loaded in {time.time() - load_start_time:.2f}s")
    return model, tokenizer

# --- 4. Prompt Engineering ---

def make_sql_prompt(question: str) -> str:
    """Creates a structured prompt for SQL generation with T-SQL specific rules."""
    return f"""### Task
Generate a T-SQL query for Azure SQL Server/SQL Server that answers the following question.

### T-SQL Rules and Requirements:
- Use proper T-SQL syntax only (no PostgreSQL, MySQL, or other SQL dialects)
- Use DATEPART(), YEAR(), MONTH(), DAY() for date functions instead of EXTRACT()
- Use ISNULL() instead of COALESCE() when possible
- Use TOP instead of LIMIT
- For pagination, use OFFSET...FETCH NEXT instead of LIMIT
- Use proper JOIN syntax with explicit INNER/LEFT/RIGHT/FULL OUTER
- For string operations, use LEN() instead of LENGTH(), CHARINDEX() instead of POSITION()
- Use GETDATE() for current datetime, not NOW()
- For conditional logic, prefer CASE WHEN over IIF() for compatibility
- Do NOT use NULLS FIRST/NULLS LAST in ORDER BY (not supported in T-SQL)
- Use proper table aliases and qualify column names where ambiguous
- For date formatting, use FORMAT() or CONVERT() functions
- Use appropriate data types: VARCHAR(MAX), NVARCHAR(MAX), DECIMAL, DATETIME2, etc.

### Schema
{DATABASE_SCHEMA}


Join Conditions:
fact_claims_dtl.claim_reference_id = dim_claims.claim_reference_id AND fact_claims_dtl.org_id = dim_claims.org_id
fct_policy.policy_number = dim_policy.policy_number AND fct_policy.org_id = dim_policy.org_id

Use the correct table based on keywords:
   - Use `dwh.fact_premium` for premium/payment-related metrics.
   - Use `dwh.dim_claims` or `dwh.fact_claims_dtl` for claim-related details.
   - Use `dwh.dim_policy` for policy metadata (start/end dates, renewals, etc.).
Use correct date fields:
   - Use `date_paid` for premium payment dates.
   - Use `date_claim_opened`, `date_closed`, etc. for claims.
   - Use `effective_date`, `expiry_date_time` for policies.

### Question
{question}

### Instructions
- Write clean, efficient T-SQL code
- Include appropriate WHERE clauses for performance
- Use meaningful table aliases (e.g., dc for dim_claims, dp for dim_policy)
- Add comments for complex logic
- Ensure all column references are valid according to the schema
- Return only the SQL query without explanations

### T-SQL Query
```sql
"""

def make_analysis_prompt(question: str, sql_query: str, sql_result: dict) -> str:
    """Creates a streamlined prompt for Phi-3-mini to analyze SQL results."""
    # Extract key data for analysis
    if sql_result.get("success") and sql_result.get("data"):
        data = sql_result["data"]
        # Simplify data representation for short, focused analysis
        if len(data) == 1 and len(data[0]) == 1:
            # Single value result - extract the value
            value = list(data[0].values())[0]
            data_summary = f"Result: {value}"
        else:
            # Multiple values - create concise summary
            data_summary = f"Data: {json.dumps(data[:3], default=str)}"  # Show first 3 rows only
            if len(data) > 3:
                data_summary += f" (showing 3 of {len(data)} rows)"
    else:
        data_summary = "No data returned or query failed"

    return f"""<|system|>
You are a concise data analyst. Provide a brief, direct answer with key insights only.
<|end|>
<|user|>
Question: {question}
{data_summary}

Provide a concise analysis in 2-3 sentences maximum.
<|end|>
<|assistant|>
"""

# --- 5. High-Performance Inference ---

@contextmanager
def inference_mode():
    """Context manager for optimized inference."""
    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                yield
        else:
            yield

def generate_text_optimized(prompt: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, max_new_tokens: int, temperature: float = 0.0) -> str:
    """Highly optimized text generation function."""
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
            'num_beams': 1 if temperature == 0 else 4,
            'pad_token_id': tokenizer.eos_token_id,
            'use_cache': True,
            'early_stopping': True,
        }
        if temperature > 0:
            gen_kwargs['temperature'] = temperature

        outputs = model.generate(**gen_kwargs)

    new_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Aggressive cleanup
    del inputs, outputs, gen_kwargs
    cleanup_memory()

    return generated_text

def extract_sql_from_response(response: str) -> str:
    """Extracts SQL code from the model's response with improved T-SQL parsing."""
    # First try to find SQL in code blocks
    if "```sql" in response:
        start = response.find("```sql") + 6
        end = response.find("```", start)
        if end != -1:
            sql_query = response[start:end].strip()
            return clean_tsql_query(sql_query)
    
    # Try to find SQL after "T-SQL Query" or similar markers
    markers = ["### T-SQL Query", "T-SQL Query:", "Query:", "SELECT", "WITH", "INSERT", "UPDATE", "DELETE"]
    for marker in markers:
        if marker in response:
            start = response.find(marker)
            if marker in ["SELECT", "WITH", "INSERT", "UPDATE", "DELETE"]:
                # For SQL keywords, include them in the result
                sql_part = response[start:]
            else:
                # For other markers, skip the marker itself
                sql_part = response[start + len(marker):]
            
            # Extract until we hit a non-SQL line or end
            lines = sql_part.split('\n')
            sql_lines = []
            for line in lines:
                cleaned_line = line.strip()
                if not cleaned_line:
                    continue
                if cleaned_line.startswith('#') or cleaned_line.startswith('--'):
                    continue
                if any(keyword in cleaned_line.upper() for keyword in ['SELECT', 'FROM', 'WHERE', 'JOIN', 'GROUP BY', 'ORDER BY', 'HAVING', 'WITH', 'INSERT', 'UPDATE', 'DELETE']) or sql_lines:
                    sql_lines.append(line.rstrip())
                elif sql_lines:
                    # Stop if we've started collecting SQL and hit a non-SQL line
                    break
            
            if sql_lines:
                return clean_tsql_query('\n'.join(sql_lines))
    
    # Fallback: return the response as-is if no SQL structure found
    return clean_tsql_query(response.strip())

def clean_tsql_query(sql_query: str) -> str:
    """Clean and validate T-SQL query for common issues."""
    if not sql_query:
        return sql_query
    
    # Remove common non-T-SQL patterns and fix them
    lines = sql_query.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.rstrip()
        if not line.strip():
            cleaned_lines.append(line)
            continue
            
        # Skip comment lines but keep SQL comments
        if line.strip().startswith('#'):
            continue
        
        # Fix common non-T-SQL patterns
        line_upper = line.upper()
        
        # Replace LIMIT with TOP (basic pattern)
        if 'LIMIT ' in line_upper and 'SELECT' in line_upper:
            # This is a simple replacement - more complex logic might be needed
            line = line.replace('LIMIT ', '-- LIMIT converted to TOP: ')
        
        # Remove NULLS FIRST/LAST
        if 'NULLS FIRST' in line_upper or 'NULLS LAST' in line_upper:
            line = line.replace('NULLS FIRST', '').replace('NULLS LAST', '')
            line = line.replace('nulls first', '').replace('nulls last', '')
        
        cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines).strip()
    
    # Final cleanup - remove trailing semicolons if multiple exist
    while result.endswith(';;'):
        result = result[:-1]
    
    return result

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
    """Generates a natural language analysis using Phi-3-mini."""
    start_time = time.time()
    prompt = make_analysis_prompt(question, sql_query, sql_result)
    analysis = generate_text_optimized(
        prompt, 
        app.state.analysis_model, 
        app.state.analysis_tokenizer, 
        max_new_tokens=150,  # Reduced for more concise responses
        temperature=0.1      # Lower temperature for focused answers
    )
    analysis_time = time.time() - start_time
    logger.info(f"Analysis generated in {analysis_time:.2f}s using Phi-3-mini")
    return analysis.strip(), analysis_time

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
        result = {"error": str(e), "status_code": getattr(e.response, 'status_code', 500) if hasattr(e, 'response') and e.response else 500}

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

    try:
        # 1. Generate SQL using SQLCoder
        sql_query, sql_generation_time = await loop.run_in_executor(
            executor, generate_sql, request.question
        )

        if not sql_query:
            raise HTTPException(status_code=400, detail="Failed to generate SQL query.")

        # 2. Execute SQL
        sql_execution_result, sql_execution_time = await loop.run_in_executor(
            executor, execute_sql, sql_query
        )

        # 3. Analyze Results using Phi-3-mini
        llm_analysis, llm_analysis_time = await loop.run_in_executor(
            executor, analyze_results, request.question, sql_query, sql_execution_result
        )

        total_processing_time = time.time() - total_start_time
        
        # Determine success based on SQL execution result
        is_successful = (
            sql_execution_result.get("success", False) or
            (sql_execution_result.get("data") is not None and "error" not in sql_execution_result)
        )
        
        response_data = {
            "success": is_successful,
            "question": request.question,
            "generated_sql": sql_query,
            "sql_execution_result": sql_execution_result,
            "llm_analysis": llm_analysis,
            "sql_generation_time": round(sql_generation_time, 2),
            "llm_analysis_time": round(llm_analysis_time, 2),
            "sql_execution_time": round(sql_execution_time, 2),
            "total_processing_time": round(total_processing_time, 2),
        }

        # 4. Save results
        file_saved = await loop.run_in_executor(
            executor, save_result_to_file, response_data
        )
        
        response_data["file_saved"] = file_saved
        return QueryResponse(**response_data)

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

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
    return {"message": "Dual-Model SQL Generation and Analysis API is running with Phi-3-mini for analysis."}

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
