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
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 1. Configuration ---

# Model and API Configuration
MODEL_NAME = "defog/llama-3-sqlcoder-8b"  # Using the newer model
API_ENDPOINT = "http://172.200.64.182:7860/execute" # API for SQL execution

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

app = FastAPI(title="Optimized SQL Query Generator and Analyzer", version="2.0.0")
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

def load_model_and_tokenizer(device: torch.device) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer with advanced optimizations."""
    logger.info("🔄 Loading model and tokenizer...")
    load_start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, padding_side="left")
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
            "load_in_8bit": True, # Use 8-bit quantization
        })

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
    
    model.eval()
    if hasattr(model, 'config'):
        model.config.use_cache = True

    logger.info(f"🤖 Model and tokenizer loaded in {time.time() - load_start_time:.2f}s")
    return model, tokenizer

# --- 4. Prompt Engineering and Caching ---

def make_sql_prompt(question: str) -> str:
    """Creates a structured prompt for SQL generation."""
    return f"""### Task
Generate a T-SQL query that answers the following question. Use the provided schema.
Ensure the query is valid for Azure SQL Server.

### Schema
{DATABASE_SCHEMA}

### Question
{question}

### SQL
```sql
"""

def make_analysis_prompt(question: str, sql_query: str, sql_result: dict) -> str:
    """Creates a prompt for analyzing SQL results."""
    # Truncate long results to fit within context window
    result_str = json.dumps(sql_result, indent=2, default=str)
    if len(result_str) > 2000:
        result_str = result_str[:2000] + "\n... (results truncated)"

    return f"""### Task
Analyze the provided SQL query and its results to answer the original question.
Provide a clear, concise, and insightful summary.

### Original Question
{question}

### Executed SQL Query
```sql
{sql_query}
