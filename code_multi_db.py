import torch
import time
import gc
import asyncio
import json
import os
import logging
import numpy as np
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import faiss
import pyodbc
from sqlalchemy import create_engine, text
import urllib

# --- 1. Configuration ---
SQL_MODEL_NAME = "defog/llama-3-sqlcoder-8b"
ANALYSIS_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SAVE_PATH = "/home/text_sql"
EMBEDDINGS_PATH = "/home/text_sql/embeddings"
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(EMBEDDINGS_PATH, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Session storage (in production, use Redis or database)
active_sessions = {}

# Schema Information as Table Chunks (unchanged from original)
TABLE_CHUNKS = [
    {
        "type": "table",
        "table": "dwh.dim_claims",
        "text": "Table: dwh.dim_claims | Columns: claim_reference_id, date_claim_first_notified, date_of_loss_from, date_claim_opened, date_of_loss_to, cause_of_loss_code, loss_description, date_coverage_confirmed, date_closed, date_claim_amount_agreed, date_paid_final_amount, date_fees_paid_final_amount, date_reopened, date_claim_denied, date_claim_withdrawn, status, refer_to_underwriters, denial_indicator, reason_for_denial, claim_total_claimed_amount, settlement_currency_code, indemnity_amount_paid, fees_amount_paid, expenses_paid_amount, dw_ins_upd_dt, org_id",
        "description": "Contains claim metadata and lifecycle events including claim status, dates, amounts, and denial information.",
        "columns": [
            "claim_reference_id", "date_claim_first_notified", "date_of_loss_from", "date_claim_opened", 
            "date_of_loss_to", "cause_of_loss_code", "loss_description", "date_coverage_confirmed", 
            "date_closed", "date_claim_amount_agreed", "date_paid_final_amount", "date_fees_paid_final_amount", 
            "date_reopened", "date_claim_denied", "date_claim_withdrawn", "status", "refer_to_underwriters", 
            "denial_indicator", "reason_for_denial", "claim_total_claimed_amount", "settlement_currency_code", 
            "indemnity_amount_paid", "fees_amount_paid", "expenses_paid_amount", "dw_ins_upd_dt", "org_id"
        ],
        "keywords": ["claims", "claim", "loss", "damage", "settlement", "denial", "status", "indemnity", "fees"]
    },
    {
        "type": "table",
        "table": "dwh.dim_policy",
        "text": "Table: dwh.dim_policy | Columns: Id, agreement_id, policy_number, new_or_renewal, group_reference, broker_reference, changed_date, effective_date, start_date_time, expiry_date_time, renewal_date_time, product_code, product_name, country_code, country, a3_country_code, country_sub_division_code, class_of_business_code, classof_business_name, main_line_of_business_name, insurance_type, section_details_number, section_details_code, section_details_name, line_of_business, section_details_description, dw_ins_upd_dt, org_id, document_id",
        "description": "Stores policy details, effective dates, product information, and geographical coverage.",
        "columns": [
            "Id", "agreement_id", "policy_number", "new_or_renewal", "group_reference", "broker_reference", 
            "changed_date", "effective_date", "start_date_time", "expiry_date_time", "renewal_date_time", 
            "product_code", "product_name", "country_code", "country", "a3_country_code", 
            "country_sub_division_code", "class_of_business_code", "classof_business_name", 
            "main_line_of_business_name", "insurance_type", "section_details_number", "section_details_code", 
            "section_details_name", "line_of_business", "section_details_description", "dw_ins_upd_dt", 
            "org_id", "document_id"
        ],
        "keywords": ["policy", "policies", "coverage", "product", "business", "renewal", "effective", "expiry"]
    },
    {
        "type": "table",
        "table": "dwh.fact_claims_dtl",
        "text": "Table: dwh.fact_claims_dtl | Columns: Id, claim_reference_id, agreement_id, policy_number, org_id, riskitems_id, Payment_Detail_Settlement_Currency_Code, Paid_Amount, Expenses_Paid_Total_Amount, Coverage_Legal_Fees_Total_Paid_Amount, Defence_Legal_Fees_Total_Paid_Amount, Adjusters_Fees_Total_Paid_Amount, TPAFees_Paid_Amount, Fees_Paid_Amount, Incurred_Detail_Settlement_Currency_Code, Indemnity_Amount, Expenses_Amount, Coverage_Legal_Fees_Amount, Defence_Fees_Amount, Adjuster_Fees_Amount, TPAFees_Amount, Fees_Amount, indemnity_reserves_amount, dw_ins_upd_dt, indemnity_amount_paid",
        "description": "Detailed claim financial information including payments, expenses, fees, and reserves.",
        "columns": [
            "Id", "claim_reference_id", "agreement_id", "policy_number", "org_id", "riskitems_id", 
            "Payment_Detail_Settlement_Currency_Code", "Paid_Amount", "Expenses_Paid_Total_Amount", 
            "Coverage_Legal_Fees_Total_Paid_Amount", "Defence_Legal_Fees_Total_Paid_Amount", 
            "Adjusters_Fees_Total_Paid_Amount", "TPAFees_Paid_Amount", "Fees_Paid_Amount", 
            "Incurred_Detail_Settlement_Currency_Code", "Indemnity_Amount", "Expenses_Amount", 
            "Coverage_Legal_Fees_Amount", "Defence_Fees_Amount", "Adjuster_Fees_Amount", 
            "TPAFees_Amount", "Fees_Amount", "indemnity_reserves_amount", "dw_ins_upd_dt", 
            "indemnity_amount_paid"
        ],
        "keywords": ["claim details", "payments", "expenses", "fees", "legal", "adjusters", "reserves", "financial"]
    },
    {
        "type": "table",
        "table": "dwh.fact_premium",
        "text": "Table: dwh.fact_premium | Columns: Id, agreement_id, policy_number, org_id, riskitems_id, original_currency_code, total_paid, instalments_amount, taxes_amount_paid, commission_percentage, commission_amount_paid, brokerage_amount_paid, insurance_amount_paid, additional_fees_paid, settlement_currency_code, gross_premium_settlement_currency, brokerage_amount_paid_settlement_currency, net_premium_settlement_currency, commission_amount_paid_settlement_currency, final_net_premium_settlement_currency, rate_of_exchange, total_settlement_amount_paid, date_paid, transaction_type, net_amount, gross_premium_paid_this_time, final_net_premium, tax_amount, dw_ins_upd_dt",
        "description": "Premium payment transactions including commissions, taxes, brokerage, and currency information.",
        "columns": [
            "Id", "agreement_id", "policy_number", "org_id", "riskitems_id", "original_currency_code", 
            "total_paid", "instalments_amount", "taxes_amount_paid", "commission_percentage", 
            "commission_amount_paid", "brokerage_amount_paid", "insurance_amount_paid", 
            "additional_fees_paid", "settlement_currency_code", "gross_premium_settlement_currency", 
            "brokerage_amount_paid_settlement_currency", "net_premium_settlement_currency", 
            "commission_amount_paid_settlement_currency", "final_net_premium_settlement_currency", 
            "rate_of_exchange", "total_settlement_amount_paid", "date_paid", "transaction_type", 
            "net_amount", "gross_premium_paid_this_time", "final_net_premium", "tax_amount", 
            "dw_ins_upd_dt"
        ],
        "keywords": ["premium", "payments", "commission", "brokerage", "taxes", "instalments", "currency"]
    },
    {
        "type": "table",
        "table": "dwh.fct_policy",
        "text": "Table: dwh.fct_policy | Columns: Id, agreement_id, policy_number, org_id, start_date, annual_premium, sum_insured, limit_of_liability, final_net_premium, tax_amount, final_net_premium_settlement_currency, settlement_currency_code, gross_premium_before_taxes_amount, dw_ins_upd_dt, document_id, gross_premium_paid_this_time",
        "description": "Policy summary information including premiums, limits, and financial aggregates.",
        "columns": [
            "Id", "agreement_id", "policy_number", "org_id", "start_date", "annual_premium", 
            "sum_insured", "limit_of_liability", "final_net_premium", "tax_amount", 
            "final_net_premium_settlement_currency", "settlement_currency_code", 
            "gross_premium_before_taxes_amount", "dw_ins_upd_dt", "document_id", 
            "gross_premium_paid_this_time"
        ],
        "keywords": ["policy summary", "annual premium", "sum insured", "liability", "limits", "aggregates"]
    }
]

JOIN_CONDITIONS = """
Join Conditions:
- fact_claims_dtl.claim_reference_id = dim_claims.claim_reference_id AND fact_claims_dtl.org_id = dim_claims.org_id
- fct_policy.policy_number = dim_policy.policy_number AND fct_policy.org_id = dim_policy.org_id
- fact_premium.policy_number = dim_policy.policy_number AND fact_premium.org_id = dim_policy.org_id

Table Usage Guidelines:
- Use `dwh.fact_premium` for premium/payment-related metrics and transactions
- Use `dwh.dim_claims` or `dwh.fact_claims_dtl` for claim-related details and financials
- Use `dwh.dim_policy` for policy metadata (start/end dates, renewals, products)
- Use `dwh.fct_policy` for policy-level financial summaries

Date Field Guidelines:
- Use `date_paid` for premium payment dates in fact_premium
- Use `date_claim_opened`, `date_closed`, etc. for claims in dim_claims
- Use `effective_date`, `expiry_date_time` for policies in dim_policy
"""

TSQL_RULES = """
T-SQL Rules and Requirements:
- Use proper T-SQL syntax only (no PostgreSQL, MySQL, or other SQL dialects)
- Use DATEPART(), YEAR(), MONTH(), DAY() for date functions instead of EXTRACT()
- Use ISNULL() instead of COALESCE() when possible
- Use TOP instead of LIMIT
- For pagination, use OFFSET...FETCH NEXT instead of LIMIT
- Use proper JOIN syntax with explicit INNER/LEFT/RIGHT/FULL OUTER
- Do NOT use the column alias in the GROUP BY or ORDER BY clauses.
- Instead, repeat the full expression used in the SELECT clause inside GROUP BY and ORDER BY
- For string operations, use LEN() instead of LENGTH(), CHARINDEX() instead of POSITION()
- Use GETDATE() for current datetime, not NOW()
- For conditional logic, prefer CASE WHEN over IIF() for compatibility
- Do NOT use NULLS FIRST/NULLS LAST in ORDER BY (not supported in T-SQL)
- Use proper table aliases and qualify column names where ambiguous
- For date formatting, use FORMAT() or CONVERT() functions
- Use appropriate data types: VARCHAR(MAX), NVARCHAR(MAX), DECIMAL, DATETIME2, etc.
- its of date time format eg:2024-11-21 06:57:57.000
"""

# --- 2. FastAPI and Pydantic Setup ---
app = FastAPI(title="RAG-Enhanced SQL Query Generator and Analyzer", version="3.0.0")
executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)

class QueryRequest(BaseModel):
    question: str
    session_id: str = None

class QueryResponse(BaseModel):
    success: bool
    question: str
    retrieved_tables: List[str]
    generated_sql: str
    sql_execution_result: dict
    llm_analysis: str
    retrieval_time: float
    sql_generation_time: float
    llm_analysis_time: float
    sql_execution_time: float
    total_processing_time: float
    file_saved: str
    retry_attempts: List[dict] = []
    total_attempts: int = 1

class SessionRequest(BaseModel):
    credentials: dict
    tables: Dict[str, bool]
    table_descriptions: Dict[str, str]
    column_descriptions: Dict[str, Dict[str, str]]

class SessionResponse(BaseModel):
    success: bool
    message: str
    session_id: str = None

# --- 3. RAG Schema Retrieval System with FAISS ---
class SchemaRetriever:
    def __init__(self, table_chunks: List[Dict], embedding_model_name: str):
        self.table_chunks = table_chunks
        self.embedding_model_name = embedding_model_name
        self.embeddings_file = os.path.join(EMBEDDINGS_PATH, "table_embeddings.npy")
        self.metadata_file = os.path.join(EMBEDDINGS_PATH, "table_metadata.json")
        self.faiss_index_file = os.path.join(EMBEDDINGS_PATH, "faiss_index.index")
        self.embeddings = None
        self.faiss_index = None
        self.embedding_model = None
        self._is_initialized = False

    def load_embedding_model(self):
        """Load the sentence transformer model for embeddings."""
        if self.embedding_model is None:
            logger.info(f"🔄 Loading embedding model: {self.embedding_model_name}")
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info("✅ Embedding model loaded successfully")

    def _initialize_if_needed(self):
        """Initialize the retriever if not already done."""
        if not self._is_initialized:
            if not self.load_embeddings():
                self.create_embeddings()
            self._is_initialized = True

    def create_embeddings(self):
        """Create embeddings for all table chunks and build a FAISS index."""
        logger.info("🔄 Creating embeddings for table chunks...")
        self.load_embedding_model()

        texts_to_embed = []
        for chunk in self.table_chunks:
            combined_text = f"{chunk['text']} {chunk['description']} {' '.join(chunk['keywords'])}"
            texts_to_embed.append(combined_text)

        embeddings = self.embedding_model.encode(texts_to_embed)
        self.embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        np.save(self.embeddings_file, self.embeddings)

        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(self.embeddings.astype('float32'))
        faiss.write_index(self.faiss_index, self.faiss_index_file)

        with open(self.metadata_file, 'w') as f:
            json.dump({
                "table_names": [chunk["table"] for chunk in self.table_chunks],
                "embedding_dim": self.embeddings.shape[1],
                "num_tables": len(self.table_chunks)
            }, f)

        logger.info(f"✅ Created and saved embeddings for {len(self.table_chunks)} tables")

    def load_embeddings(self):
        """Load existing embeddings and FAISS index from files."""
        try:
            if (os.path.exists(self.embeddings_file) and 
                os.path.exists(self.metadata_file) and 
                os.path.exists(self.faiss_index_file)):
                
                self.embeddings = np.load(self.embeddings_file)
                self.faiss_index = faiss.read_index(self.faiss_index_file)
                
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                logger.info(f"✅ Loaded embeddings and FAISS index for {metadata['num_tables']} tables")
                return True
        except Exception as e:
            logger.warning(f"Failed to load existing embeddings: {e}")
            for file_path in [self.embeddings_file, self.metadata_file, self.faiss_index_file]:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except:
                        pass
        return False

    def retrieve_relevant_tables(self, question: str, top_k: int = 3) -> List[Dict]:
        """Retrieve top-k most relevant tables using FAISS."""
        self._initialize_if_needed()
        self.load_embedding_model()

        question_embedding = self.embedding_model.encode([question])
        question_embedding_normalized = question_embedding / np.linalg.norm(question_embedding, axis=1, keepdims=True)

        try:
            D, I = self.faiss_index.search(question_embedding_normalized.astype('float32'), k=min(top_k, len(self.table_chunks)))
            relevant_tables = [self.table_chunks[i] for i in I[0] if i < len(self.table_chunks)]
            logger.info(f"Retrieved {len(relevant_tables)} relevant tables: {[t['table'] for t in relevant_tables]}")
            return relevant_tables
        except Exception as e:
            logger.error(f"Error during FAISS search: {e}")
            return self._fallback_retrieval(question, top_k)

    def _fallback_retrieval(self, question: str, top_k: int = 3) -> List[Dict]:
        """Fallback retrieval method using simple keyword matching."""
        logger.info("Using fallback keyword-based retrieval")
        question_lower = question.lower()
        scores = []
        
        for i, chunk in enumerate(self.table_chunks):
            score = 0
            for keyword in chunk['keywords']:
                if keyword.lower() in question_lower:
                    score += 2
            table_name_parts = chunk['table'].split('.')[-1].split('_')
            for part in table_name_parts:
                if part.lower() in question_lower:
                    score += 1
            desc_words = chunk['description'].lower().split()
            for word in desc_words:
                if len(word) > 3 and word in question_lower:
                    score += 0.5
            scores.append((score, i))
        
        scores.sort(reverse=True, key=lambda x: x[0])
        relevant_tables = [self.table_chunks[i] for _, i in scores[:top_k]]
        logger.info(f"Fallback retrieved {len(relevant_tables)} tables: {[t['table'] for t in relevant_tables]}")
        return relevant_tables

def construct_rag_prompt(question: str, relevant_tables: List[Dict]) -> str:
    """Creates a structured prompt using retrieved schema elements."""
    schema_section = ""
    for table_info in relevant_tables:
        schema_section += f"""
TABLE: {table_info['table']}
Description: {table_info['description']}
Columns: {', '.join(table_info['columns'])}
"""
    
    prompt = f"""### Task
Generate a T-SQL query for Azure SQL Server/SQL Server that answers the following question using only the provided relevant schema information.

### Retrieved Schema Information
{schema_section}
{JOIN_CONDITIONS}

{TSQL_RULES}

### Question
{question}

### Instructions
- Use ONLY the tables and columns provided in the retrieved schema above
- Write clean, efficient T-SQL code with appropriate WHERE clauses for performance
- Use meaningful table aliases (e.g., dc for dim_claims, dp for dim_policy)
- Return ONLY the SQL query without any explanations, comments, or additional text
- Do not include any markdown formatting or code block markers
- Do not provide any analysis or explanation after the query

### T-SQL Query:
"""
    return prompt

def construct_retry_prompt(question: str, relevant_tables: List[Dict], previous_sql: str, error_message: str, attempt_number: int) -> str:
    """Creates a retry prompt that includes the previous SQL error for correction."""
    schema_section = ""
    for table_info in relevant_tables:
        schema_section += f"""
TABLE: {table_info['table']}
Description: {table_info['description']}
Columns: {', '.join(table_info['columns'])}
"""
    
    prompt = f"""### Task
Generate a corrected T-SQL query for Azure SQL Server/SQL Server. The previous attempt failed with an error.

### Retrieved Schema Information
{schema_section}
{JOIN_CONDITIONS}

{TSQL_RULES}

### Question
{question}

### Previous Attempt (Failed)
SQL Query:
{previous_sql}

Error Message:
{error_message}

### Instructions for Retry (Attempt {attempt_number + 1})
- Analyze the error message and fix the specific issue in the previous SQL
- Use ONLY the tables and columns provided in the retrieved schema above
- Write clean, efficient T-SQL code with appropriate WHERE clauses for performance
- Use meaningful table aliases (e.g., dc for dim_claims, dp for dim_policy)
- Pay special attention to:
  * Column name spelling and existence
  * Proper JOIN conditions
  * Date format requirements
  * T-SQL syntax compliance
- Return ONLY the corrected SQL query without any explanations, comments, or additional text
- Do not include any markdown formatting or code block markers

### Corrected T-SQL Query:
"""
    return prompt

# --- 4. Device and Model Loading ---
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

# --- 5. Core Logic Functions ---
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
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device, non_blocking=True) for k, v in inputs.items()}
    input_length = inputs['input_ids'].shape[1]
    with inference_mode():
        gen_kwargs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs.get('attention_mask'),
            'max_new_tokens': max_new_tokens,
            'do_sample': False,
            'num_beams': 1,
            'pad_token_id': tokenizer.eos_token_id,
            'use_cache': False if "phi-3" in model.config.model_type.lower() else True,
        }
        if temperature > 0:
            gen_kwargs['temperature'] = temperature
            gen_kwargs['do_sample'] = True
        
        try:
            outputs = model.generate(**gen_kwargs)
        except AttributeError as e:
            if "get_max_length" in str(e):
                gen_kwargs['use_cache'] = False
                outputs = model.generate(**gen_kwargs)
            else:
                raise
    
    new_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    del inputs, outputs, gen_kwargs
    cleanup_memory()
    return generated_text

def extract_sql_from_response(response: str) -> str:
    """Extracts SQL code from the model's response with improved T-SQL parsing."""
    if "```sql" in response:
        start = response.find("```sql") + 6
        end = response.find("```", start)
        if end != -1:
            sql_query = response[start:end].strip()
            return clean_tsql_query(sql_query)
    markers = ["### T-SQL Query", "T-SQL Query:", "Query:", "SELECT", "WITH", "INSERT", "UPDATE", "DELETE"]
    for marker in markers:
        if marker in response:
            start = response.find(marker)
            if marker in ["SELECT", "WITH", "INSERT", "UPDATE", "DELETE"]:
                sql_part = response[start:]
            else:
                sql_part = response[start + len(marker):]
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
                    break
            if sql_lines:
                return clean_tsql_query('\n'.join(sql_lines))
    return clean_tsql_query(response.strip())

def clean_tsql_query(sql_query: str) -> str:
    """Clean and validate T-SQL query for common issues."""
    if not sql_query:
        return sql_query
    lines = sql_query.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.rstrip()
        if not line.strip():
            cleaned_lines.append(line)
            continue
        if line.strip().startswith('#'):
            continue
        line_upper = line.upper()
        if 'LIMIT ' in line_upper and 'SELECT' in line_upper:
            line = line.replace('LIMIT ', '-- LIMIT converted to TOP: ')
        if 'NULLS FIRST' in line_upper or 'NULLS LAST' in line_upper:
            line = line.replace('NULLS FIRST', '').replace('NULLS LAST', '')
            line = line.replace('nulls first', '').replace('nulls last', '')
        cleaned_lines.append(line)
    result = '\n'.join(cleaned_lines).strip()
    while result.endswith(';;'):
        result = result[:-1]
    return result

def execute_sql_direct(sql_query: str, connection_string: str) -> Tuple[Dict[str, Any], float]:
    """Executes SQL query directly using the provided connection string."""
    start_time = time.time()
    try:
        engine = create_engine(connection_string)
        
        with engine.connect() as connection:
            result = connection.execute(text(sql_query))
            rows = result.fetchall()
            columns = result.keys()
            data = [dict(zip(columns, row)) for row in rows]
            
            execution_result = {
                "success": True,
                "data": data,
                "row_count": len(data)
            }
            
    except Exception as e:
        logger.error(f"SQL execution error: {e}")
        execution_result = {
            "success": False,
            "error": str(e),
            "data": None
        }
    
    execution_time = time.time() - start_time
    logger.info(f"SQL executed in {execution_time:.2f}s")
    return execution_result, execution_time

def build_connection_string(credentials: dict) -> str:
    """Builds SQL Server connection string from credentials."""
    try:
        if credentials.get('driver', 'ODBC Driver 17 for SQL Server'):
            conn_str = (
                f"DRIVER={{{credentials.get('driver', 'ODBC Driver 17 for SQL Server')}}};"
                f"SERVER={credentials['server']};"
                f"DATABASE={credentials['database']};"
            )
            
            if credentials.get('username') and credentials.get('password'):
                conn_str += f"UID={credentials['username']};PWD={credentials['password']};"
            else:
                conn_str += "Trusted_Connection=yes;"
            
            params = urllib.parse.quote_plus(conn_str)
            return f"mssql+pyodbc:///?odbc_connect={params}"
        
    except KeyError as e:
        raise ValueError(f"Missing required credential: {e}")

def create_dynamic_table_chunks(tables: Dict[str, bool], table_descriptions: Dict[str, str], 
                               column_descriptions: Dict[str, Dict[str, str]]) -> List[Dict]:
    """Creates table chunks based on user selection."""
    dynamic_chunks = []
    
    for table_name, is_enabled in tables.items():
        if not is_enabled:
            continue
            
        description = table_descriptions.get(table_name, "No description available")
        columns = list(column_descriptions.get(table_name, {}).keys())
        keywords = []
        keywords.extend(table_name.lower().split('_'))
        keywords.extend(description.lower().split())
        keywords = list(set([k for k in keywords if len(k) > 2]))
        
        chunk = {
            "type": "table",
            "table": table_name,
            "text": f"Table: {table_name} | Columns: {', '.join(columns)}",
            "description": description,
            "columns": columns,
            "keywords": keywords
        }
        
        dynamic_chunks.append(chunk)
    
    return dynamic_chunks

def generate_sql_with_rag_with_retry_session(question: str, retriever: SchemaRetriever, 
                                           connection_string: str, max_retries: int = 2) -> Tuple[str, List[str], float, float, List[dict]]:
    """Generates SQL query using RAG approach with error-based retry logic and direct execution."""
    retrieval_start_time = time.time()
    relevant_tables = retriever.retrieve_relevant_tables(question, top_k=3)
    retrieved_table_names = [table['table'] for table in relevant_tables]
    retrieval_time = time.time() - retrieval_start_time
    
    retry_attempts = []
    total_sql_generation_time = 0.0
    
    for attempt in range(max_retries + 1):
        sql_generation_start_time = time.time()
        
        if attempt == 0:
            prompt = construct_rag_prompt(question, relevant_tables)
        else:
            previous_error = retry_attempts[-1]['error']
            previous_sql = retry_attempts[-1]['sql_query']
            prompt = construct_retry_prompt(question, relevant_tables, previous_sql, previous_error, attempt)
        
        response = generate_text_optimized(prompt, app.state.sql_model, app.state.sql_tokenizer, max_new_tokens=300)
        sql_query = extract_sql_from_response(response)
        
        attempt_sql_generation_time = time.time() - sql_generation_start_time
        total_sql_generation_time += attempt_sql_generation_time
        
        sql_result, _ = execute_sql_direct(sql_query, connection_string)
        
        if sql_result.get("success") or (sql_result.get("data") is not None and "error" not in sql_result):
            logger.info(f"SQL generated successfully on attempt {attempt + 1}")
            logger.info(f"Schema retrieval completed in {retrieval_time:.2f}s using tables: {retrieved_table_names}")
            logger.info(f"Total SQL generation time: {total_sql_generation_time:.2f}s across {attempt + 1} attempts")
            return sql_query, retrieved_table_names, retrieval_time, total_sql_generation_time, retry_attempts
        else:
            error_message = sql_result.get("error", "Unknown SQL execution error")
            retry_info = {
                "attempt": attempt + 1,
                "sql_query": sql_query,
                "error": error_message,
                "generation_time": attempt_sql_generation_time
            }
            retry_attempts.append(retry_info)
            
            logger.warning(f"SQL attempt {attempt + 1} failed: {error_message}")
            
            if attempt == max_retries:
                logger.error(f"All {max_retries + 1} SQL generation attempts failed")
                logger.info(f"Schema retrieval completed in {retrieval_time:.2f}s using tables: {retrieved_table_names}")
                logger.info(f"Total SQL generation time: {total_sql_generation_time:.2f}s across {attempt + 1} attempts")
                return sql_query, retrieved_table_names, retrieval_time, total_sql_generation_time, retry_attempts
    
    return sql_query, retrieved_table_names, retrieval_time, total_sql_generation_time, retry_attempts

def make_analysis_prompt(question: str, sql_query: str, sql_result: dict) -> str:
    """Creates a streamlined prompt for Phi-3-mini to analyze SQL results."""
    if sql_result.get("success") and sql_result.get("data"):
        data = sql_result["data"]
        if len(data) == 1 and len(data[0]) == 1:
            value = list(data[0].values())[0]
            data_summary = f"Result: {value}"
        else:
            data_summary = f"Data: {json.dumps(data[:3], default=str)}"
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

def analyze_results(question: str, sql_query: str, sql_result: dict) -> Tuple[str, float]:
    """Generates a natural language analysis using Phi-3-mini."""
    start_time = time.time()
    prompt = make_analysis_prompt(question, sql_query, sql_result)
    analysis = generate_text_optimized(
        prompt, 
        app.state.analysis_model, 
        app.state.analysis_tokenizer, 
        max_new_tokens=150,
        temperature=0.1
    )
    analysis_time = time.time() - start_time
    logger.info(f"Analysis generated in {analysis_time:.2f}s using Phi-3-mini")
    return analysis.strip(), analysis_time

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

# --- 6. FastAPI Application Events and Endpoints ---
@app.on_event("startup")
async def startup_event():
    """Load both models on application startup."""
    app.state.device = setup_device()
    app.state.sql_model, app.state.sql_tokenizer = load_sql_model_and_tokenizer(app.state.device)
    app.state.analysis_model, app.state.analysis_tokenizer = load_analysis_model_and_tokenizer(app.state.device)
    app.state.loop = asyncio.get_event_loop()
    logger.info("🚀 Models loaded successfully!")

@app.on_event("shutdown")
def shutdown_event():
    """Clean up resources on shutdown."""
    cleanup_memory()
    executor.shutdown(wait=True)
    logger.info("🧹 Memory cleaned up and executor shut down.")

@app.post("/start-session", response_model=SessionResponse)
async def start_session(request: SessionRequest):
    """Start a new session with connection string and table selection."""
    try:
        connection_string = build_connection_string(request.credentials)
        
        try:
            engine = create_engine(connection_string)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Database connection failed: {str(e)}")
        
        import uuid
        session_id = str(uuid.uuid4())
        
        table_chunks = create_dynamic_table_chunks(
            request.tables, 
            request.table_descriptions, 
            request.column_descriptions
        )
        
        if not table_chunks:
            raise HTTPException(status_code=400, detail="No tables selected")
        
        active_sessions[session_id] = {
            "connection_string": connection_string,
            "table_chunks": table_chunks,
            "created_at": time.time()
        }
        
        return SessionResponse(
            success=True,
            message=f"Session started successfully with {len(table_chunks)} tables",
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"Session start error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start session: {str(e)}")

@app.post("/generate-and-analyze-sql", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Main endpoint to generate SQL, execute it, and analyze results with session support."""
    if not request.session_id or request.session_id not in active_sessions:
        raise HTTPException(status_code=400, detail="Invalid or missing session_id. Please start a session first.")
    
    session_data = active_sessions[request.session_id]
    connection_string = session_data["connection_string"]
    table_chunks = session_data["table_chunks"]
    
    total_start_time = time.time()
    loop = app.state.loop
    
    try:
        temp_retriever = SchemaRetriever(table_chunks, EMBEDDING_MODEL_NAME)
        
        sql_query, retrieved_tables, retrieval_time, sql_generation_time, retry_attempts = await loop.run_in_executor(
            executor, generate_sql_with_rag_with_retry_session, request.question, temp_retriever, connection_string, 2
        )
        
        if not sql_query:
            raise HTTPException(status_code=400, detail="Failed to generate SQL query after all attempts.")
        
        sql_execution_result, sql_execution_time = await loop.run_in_executor(
            executor, execute_sql_direct, sql_query, connection_string
        )
        
        llm_analysis, llm_analysis_time = await loop.run_in_executor(
            executor, analyze_results, request.question, sql_query, sql_execution_result
        )
        
        total_processing_time = time.time() - total_start_time
        
        is_successful = (
            sql_execution_result.get("success", False) or
            (sql_execution_result.get("data") is not None and "error" not in sql_execution_result)
        )
        
        response_data = {
            "success": is_successful,
            "question": request.question,
            "retrieved_tables": retrieved_tables,
            "generated_sql": sql_query,
            "sql_execution_result": sql_execution_result,
            "llm_analysis": llm_analysis,
            "retrieval_time": round(retrieval_time, 2),
            "sql_generation_time": round(sql_generation_time, 2),
            "llm_analysis_time": round(llm_analysis_time, 2),
            "sql_execution_time": round(sql_execution_time, 2),
            "total_processing_time": round(total_processing_time, 2),
            "retry_attempts": retry_attempts,
            "total_attempts": len(retry_attempts) + 1 if not retry_attempts or not is_successful else 1
        }
        
        file_saved = await loop.run_in_executor(executor, save_result_to_file, response_data)
        response_data["file_saved"] = file_saved
        
        return QueryResponse(**response_data)
        
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.post("/generate-sql-only")
async def generate_sql_only(request: QueryRequest):
    """Generate SQL query without execution or analysis using RAG with retry logic."""
    if not request.session_id or request.session_id not in active_sessions:
        raise HTTPException(status_code=400, detail="Invalid or missing session_id. Please start a session first.")
    
    session_data = active_sessions[request.session_id]
    connection_string = session_data["connection_string"]
    table_chunks = session_data["table_chunks"]
    
    try:
        temp_retriever = SchemaRetriever(table_chunks, EMBEDDING_MODEL_NAME)
        sql_query, retrieved_tables, retrieval_time, sql_generation_time, retry_attempts = await app.state.loop.run_in_executor(
            executor, generate_sql_with_rag_with_retry_session, request.question, temp_retriever, connection_string, 2
        )
        return {
            "question": request.question,
            "retrieved_tables": retrieved_tables,
            "generated_sql": sql_query,
            "retrieval_time": round(retrieval_time, 2),
            "sql_generation_time": round(sql_generation_time, 2),
            "total_generation_time": round(retrieval_time + sql_generation_time, 2),
            "retry_attempts": retry_attempts,
            "total_attempts": len(retry_attempts) + 1 if retry_attempts else 1
        }
    except Exception as e:
        logger.error(f"SQL generation error: {e}")
        raise HTTPException(status_code=500, detail=f"SQL generation failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "sql_model_loaded": hasattr(app.state, 'sql_model'),
        "analysis_model_loaded": hasattr(app.state, 'analysis_model'),
        "device": str(app.state.device) if hasattr(app.state, 'device') else "unknown"
    }

@app.get("/list-sessions")
async def list_sessions():
    """List all active sessions."""
    sessions = []
    current_time = time.time()
    
    for session_id, session_data in active_sessions.items():
        sessions.append({
            "session_id": session_id,
            "created_at": session_data["created_at"],
            "age_minutes": round((current_time - session_data["created_at"]) / 60, 2),
            "table_count": len(session_data["table_chunks"])
        })
    
    return {"active_sessions": sessions}

@app.delete("/end-session/{session_id}")
async def end_session(session_id: str):
    """End a specific session."""
    if session_id in active_sessions:
        del active_sessions[session_id]
        return {"message": f"Session {session_id} ended successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

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
        "embedding_model": EMBEDDING_MODEL_NAME,
        "sql_model_loaded": hasattr(app.state, 'sql_model'),
        "analysis_model_loaded": hasattr(app.state, 'analysis_model'),
        "rag_system_loaded": hasattr(app.state, 'retriever'),
    }

@app.get("/schema-info")
async def schema_info():
    """Get information about the available schema tables."""
    return {
        "total_tables": len(TABLE_CHUNKS),
        "tables": [chunk["table"] for chunk in TABLE_CHUNKS],
        "embeddings_created": os.path.exists(os.path.join(EMBEDDINGS_PATH, "table_embeddings.npy"))
    }

@app.post("/test-retrieval")
async def test_retrieval(request: QueryRequest):
    """Test the RAG retrieval system independently."""
    if not request.session_id or request.session_id not in active_sessions:
        raise HTTPException(status_code=400, detail="Invalid or missing session_id. Please start a session first.")
    
    session_data = active_sessions[request.session_id]
    table_chunks = session_data["table_chunks"]
    
    try:
        temp_retriever = SchemaRetriever(table_chunks, EMBEDDING_MODEL_NAME)
        relevant_tables = temp_retriever.retrieve_relevant_tables(request.question, top_k=3)
        return {
            "question": request.question,
            "retrieved_tables": [
                {
                    "table": table["table"],
                    "description": table["description"],
                    "keywords": table["keywords"]
                }
                for table in relevant_tables
            ]
        }
    except Exception as e:
        logger.error(f"Retrieval test error: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval test failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)