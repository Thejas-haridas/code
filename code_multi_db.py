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
from typing import Tuple, Dict, List, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import faiss  # Import FAISS for efficient similarity search

# --- 1. Configuration ---
SQL_MODEL_NAME = "defog/llama-3-sqlcoder-8b"
ANALYSIS_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"  # For analysis
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # For embeddings
SAVE_PATH = "/home/text_sql"
EMBEDDINGS_PATH = "/home/text_sql/embeddings"
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(EMBEDDINGS_PATH, exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PLATFORM_INFO = {
    "sqlserver": {
        "name": "SQL Server",
        "dialect_rules": """
- Use T-SQL syntax only
- Use ISNULL() instead of COALESCE()
- Use TOP instead of LIMIT
- Use GETDATE() for current datetime
- Use DATEPART(), YEAR(), MONTH(), DAY() for date functions
- Use LEN() instead of LENGTH()
        """,
        "connection_template": {
            "host": "localhost",
            "port": 1433,
            "username": "",
            "password": "",
            "database": "",
            "driver": "{ODBC Driver 17 for SQL Server}"
        },
        "schema": {}  # Will be populated from TABLE_CHUNKS
    },
    "oracle": {
        "name": "Oracle",
        "dialect_rules": """
- Use Oracle PL/SQL syntax
- Use NVL() instead of COALESCE()
- Use ROWNUM instead of LIMIT
- Use SYSDATE for current datetime
- Use LENGTH() instead of LEN()
- Use TO_CHAR(), TO_DATE() for date formatting
        """,
        "connection_template": {
            "dsn": "",
            "user": "",
            "password": ""
        },
        "schema": {}
    },
    "snowflake": {
        "name": "Snowflake",
        "dialect_rules": """
- Use Snowflake SQL syntax
- Use IFF() instead of IFNULL()/ISNULL()
- Use LIMIT clause
- Use CURRENT_TIMESTAMP for current datetime
- Use ARRAY_AGG() for aggregation
- Support semi-structured data types like VARIANT
        """,
        "connection_template": {
            "account": "",
            "user": "",
            "password": "",
            "warehouse": "",
            "database": "",
            "schema": ""
        },
        "schema": {}
    },
    "bigquery": {
        "name": "BigQuery",
        "dialect_rules": """
- Use BigQuery Standard SQL
- Use IFNULL() instead of COALESCE()
- Use LIMIT clause
- Use CURRENT_TIMESTAMP() for current datetime
- Use EXTRACT() for date parts
- Support nested and repeated fields
- Prefer CTEs over subqueries for clarity
        """,
        "connection_template": {
            "project_id": "",
            "dataset_id": "",
            "table_id": "",
            "credentials_path": ""
        },
        "schema": {}
    }
}
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

TABLE_CHUNKS = [
    {
        "type": "table",
        "table": "dwh.dim_claims",
        "text": "Table: dwh.dim_claims | Columns: claim_reference_id, date_claim_first_notified, date_of_loss_from, date_claim_opened, date_of_loss_to, cause_of_loss_code, loss_description, date_coverage_confirmed, date_closed, date_claim_amount_agreed, date_paid_final_amount, date_fees_paid_final_amount, date_reopened, date_claim_denied, date_claim_withdrawn, status, refer_to_underwriters, denial_indicator, reason_for_denial, claim_total_claimed_amount, settlement_currency_code, indemnity_amount_paid, fees_amount_paid, expenses_paid_amount, dw_ins_upd_dt, org_id",
        "description": "Contains claim metadata and lifecycle events including claim status, dates, amounts, and denial information.",
        "columns": ["claim_reference_id", "date_claim_first_notified", "date_of_loss_from", "date_claim_opened", "date_of_loss_to", "cause_of_loss_code", "loss_description", "date_coverage_confirmed", "date_closed", "date_claim_amount_agreed", "date_paid_final_amount", "date_fees_paid_final_amount", "date_reopened", "date_claim_denied", "date_claim_withdrawn", "status", "refer_to_underwriters", "denial_indicator", "reason_for_denial", "claim_total_claimed_amount", "settlement_currency_code", "indemnity_amount_paid", "fees_amount_paid", "expenses_paid_amount", "dw_ins_upd_dt", "org_id"],
        "keywords": ["claims", "claim", "loss", "damage", "settlement", "denial", "status", "indemnity", "fees"]
    },
    {
        "type": "table",
        "table": "dwh.dim_policy",
        "text": "Table: dwh.dim_policy | Columns: Id, agreement_id, policy_number, new_or_renewal, group_reference, broker_reference, changed_date, effective_date, start_date_time, expiry_date_time, renewal_date_time, product_code, product_name, country_code, country, a3_country_code, country_sub_division_code, class_of_business_code, classof_business_name, main_line_of_business_name, insurance_type, section_details_number, section_details_code, section_details_name, line_of_business, section_details_description, dw_ins_upd_dt, org_id, document_id",
        "description": "Stores policy details, effective dates, product information, and geographical coverage.",
        "columns": ["Id", "agreement_id", "policy_number", "new_or_renewal", "group_reference", "broker_reference", "changed_date", "effective_date", "start_date_time", "expiry_date_time", "renewal_date_time", "product_code", "product_name", "country_code", "country", "a3_country_code", "country_sub_division_code", "class_of_business_code", "classof_business_name", "main_line_of_business_name", "insurance_type", "section_details_number", "section_details_code", "section_details_name", "line_of_business", "section_details_description", "dw_ins_upd_dt", "org_id", "document_id"],
        "keywords": ["policy", "policies", "coverage", "product", "business", "renewal", "effective", "expiry"]
    },
    {
        "type": "table",
        "table": "dwh.fact_claims_dtl",
        "text": "Table: dwh.fact_claims_dtl | Columns: Id, claim_reference_id, agreement_id, policy_number, org_id, riskitems_id, Payment_Detail_Settlement_Currency_Code, Paid_Amount, Expenses_Paid_Total_Amount, Coverage_Legal_Fees_Total_Paid_Amount, Defence_Legal_Fees_Total_Paid_Amount, Adjusters_Fees_Total_Paid_Amount, TPAFees_Paid_Amount, Fees_Paid_Amount, Incurred_Detail_Settlement_Currency_Code, Indemnity_Amount, Expenses_Amount, Coverage_Legal_Fees_Amount, Defence_Fees_Amount, Adjuster_Fees_Amount, TPAFees_Amount, Fees_Amount, indemnity_reserves_amount, dw_ins_upd_dt, indemnity_amount_paid",
        "description": "Detailed claim financial information including payments, expenses, fees, and reserves.",
        "columns": ["Id", "claim_reference_id", "agreement_id", "policy_number", "org_id", "riskitems_id", "Payment_Detail_Settlement_Currency_Code", "Paid_Amount", "Expenses_Paid_Total_Amount", "Coverage_Legal_Fees_Total_Paid_Amount", "Defence_Legal_Fees_Total_Paid_Amount", "Adjusters_Fees_Total_Paid_Amount", "TPAFees_Paid_Amount", "Fees_Paid_Amount", "Incurred_Detail_Settlement_Currency_Code", "Indemnity_Amount", "Expenses_Amount", "Coverage_Legal_Fees_Amount", "Defence_Fees_Amount", "Adjuster_Fees_Amount", "TPAFees_Amount", "Fees_Amount", "indemnity_reserves_amount", "dw_ins_upd_dt", "indemnity_amount_paid"],
        "keywords": ["claim details", "payments", "expenses", "fees", "legal", "adjusters", "reserves", "financial"]
    },
    {
        "type": "table",
        "table": "dwh.fact_premium",
        "text": "Table: dwh.fact_premium | Columns: Id, agreement_id, policy_number, org_id, riskitems_id, original_currency_code, total_paid, instalments_amount, taxes_amount_paid, commission_percentage, commission_amount_paid, brokerage_amount_paid, insurance_amount_paid, additional_fees_paid, settlement_currency_code, gross_premium_settlement_currency, brokerage_amount_paid_settlement_currency, net_premium_settlement_currency, commission_amount_paid_settlement_currency, final_net_premium_settlement_currency, rate_of_exchange, total_settlement_amount_paid, date_paid, transaction_type, net_amount, gross_premium_paid_this_time, final_net_premium, tax_amount, dw_ins_upd_dt",
        "description": "Premium payment transactions including commissions, taxes, brokerage, and currency information.",
        "columns": ["Id", "agreement_id", "policy_number", "org_id", "riskitems_id", "original_currency_code", "total_paid", "instalments_amount", "taxes_amount_paid", "commission_percentage", "commission_amount_paid", "brokerage_amount_paid", "insurance_amount_paid", "additional_fees_paid", "settlement_currency_code", "gross_premium_settlement_currency", "brokerage_amount_paid_settlement_currency", "net_premium_settlement_currency", "commission_amount_paid_settlement_currency", "final_net_premium_settlement_currency", "rate_of_exchange", "total_settlement_amount_paid", "date_paid", "transaction_type", "net_amount", "gross_premium_paid_this_time", "final_net_premium", "tax_amount", "dw_ins_upd_dt"],
        "keywords": ["premium", "payments", "commission", "brokerage", "taxes", "instalments", "currency"]
    },
    {
        "type": "table",
        "table": "dwh.fct_policy",
        "text": "Table: dwh.fct_policy | Columns: Id, agreement_id, policy_number, org_id, start_date, annual_premium, sum_insured, limit_of_liability, final_net_premium, tax_amount, final_net_premium_settlement_currency, settlement_currency_code, gross_premium_before_taxes_amount, dw_ins_upd_dt, document_id, gross_premium_paid_this_time",
        "description": "Policy summary information including premiums, limits, and financial aggregates.",
        "columns": ["Id", "agreement_id", "policy_number", "org_id", "start_date", "annual_premium", "sum_insured", "limit_of_liability", "final_net_premium", "tax_amount", "final_net_premium_settlement_currency", "settlement_currency_code", "gross_premium_before_taxes_amount", "dw_ins_upd_dt", "document_id", "gross_premium_paid_this_time"],
        "keywords": ["policy summary", "annual premium", "sum insured", "liability", "limits", "aggregates"]
    }
]

# --- 2. FastAPI and Pydantic Setup ---
app = FastAPI(title="RAG-Enhanced SQL Query Generator and Analyzer", version="3.0.0")
executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)

class QueryRequest(BaseModel):
    question: str
    db_platform: str  # One of ["sqlserver", "oracle", "snowflake", "bigquery"]

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

# --- 3. RAG Schema Retrieval System with FAISS (FIXED) ---
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
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(self.embeddings.astype('float32'))
        np.save(self.embeddings_file, self.embeddings)
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

def construct_rag_prompt(question: str, relevant_tables: List[Dict], db_platform: str) -> str:
    """Creates a structured prompt using retrieved schema elements and DB dialect."""
    schema_section = ""
    for table_info in relevant_tables:
        schema_section += f"""
TABLE: {table_info['table']}
Description: {table_info['description']}
Columns: {', '.join(table_info['columns'])}
"""
    dialect_rules = DB_PLATFORMS.get(db_platform.lower(), DB_PLATFORMS["sqlserver"])
    
    prompt = f"""### Task
Generate a SQL query compatible with {dialect_rules['name']} that answers the following question using only the provided schema information.
### Retrieved Schema Information
{schema_section}
{JOIN_CONDITIONS}
{dialect_rules['rules']}
### Question
{question}
### Instructions
- Use ONLY the tables and columns provided in the retrieved schema above
- Write clean, efficient SQL code appropriate for {dialect_rules['name']}
- Use meaningful table aliases
- Return ONLY the SQL query without any explanations, comments, or additional text
- Do not include any markdown formatting or code block markers
### SQL Query:
"""
    return prompt

# --- Device and Model Loading ---
def setup_device() -> torch.device:
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

# --- Core Logic Functions ---
@contextmanager
def inference_mode():
    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                yield
        else:
            yield

def generate_text_optimized(prompt: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, max_new_tokens: int, temperature: float = 0.0) -> str:
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
    if "```sql" in response:
        start = response.find("```sql") + 6
        end = response.find("```", start)
        if end != -1:
            sql_query = response[start:end].strip()
            return clean_sql_query(sql_query)
    return clean_sql_query(response.strip())

def clean_sql_query(sql_query: str) -> str:
    lines = sql_query.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.rstrip()
        if line.strip().startswith('#') or line.strip().startswith('--'):
            continue
        cleaned_lines.append(line)
    result = '\n'.join(cleaned_lines).strip()
    while result.endswith(';;'):
        result = result[:-1]
    return result

def generate_sql_with_rag(question: str, retriever: SchemaRetriever, db_platform: str) -> Tuple[str, List[str], float, float]:
    retrieval_start_time = time.time()
    relevant_tables = retriever.retrieve_relevant_tables(question, top_k=3)
    retrieved_table_names = [table['table'] for table in relevant_tables]
    retrieval_time = time.time() - retrieval_start_time

    sql_generation_start_time = time.time()
    prompt = construct_rag_prompt(question, relevant_tables, db_platform)
    response = generate_text_optimized(prompt, app.state.sql_model, app.state.sql_tokenizer, max_new_tokens=300)
    sql_query = extract_sql_from_response(response)
    sql_generation_time = time.time() - sql_generation_start_time

    logger.info(f"Schema retrieval completed in {retrieval_time:.2f}s using tables: {retrieved_table_names}")
    logger.info(f"SQL generation completed in {sql_generation_time:.2f}s")
    return sql_query, retrieved_table_names, retrieval_time, sql_generation_time

def make_analysis_prompt(question: str, sql_query: str, sql_result: dict) -> str:
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

def execute_sql(sql_query: str) -> Tuple[Dict[str, Any], float]:
    start_time = time.time()
    result = {"data": [{"sample_column": "mock_data"}]}
    execution_time = time.time() - start_time
    logger.info(f"SQL executed in {execution_time:.2f}s")
    return result, execution_time

def save_result_to_file(data: dict) -> str:
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
