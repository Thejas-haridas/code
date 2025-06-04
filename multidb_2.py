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
from typing import Tuple, Dict, Any, List , Optional , Union
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModel


# --- 1. Configuration ---
# Model Configuration
SQL_MODEL_NAME = "defog/llama-3-sqlcoder-8b"  # For SQL generation
ANALYSIS_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"  # For analysis
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # For embeddings
API_ENDPOINT = "http://172.200.64.182:7860/execute"  # API for SQL execution

# File Path Configuration for saving results
SAVE_PATH = "/home/text_sql"
EMBEDDINGS_PATH = "/home/text_sql/embeddings"
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(EMBEDDINGS_PATH, exist_ok=True)

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced Table Schema Information (Combined from both files)
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

# Enhanced Table Descriptions (from first file)
TABLE_DESCRIPTIONS = {
    "dwh.dim_claims": {
        "description": "Contains claim metadata and lifecycle events including claim status, dates, amounts, and denial information.",
        "columns": {
            "claim_reference_id": "Unique identifier for each claim",
            "date_claim_first_notified": "Date when the claim was first reported",
            "date_of_loss_from": "Start date of the loss event",
            "date_of_loss_to": "End date of the loss event",
            "cause_of_loss_code": "Code identifying the cause of loss",
            "loss_description": "Detailed description of the loss event",
            "date_coverage_confirmed": "Date when coverage was confirmed",
            "date_closed": "Date when the claim was closed",
            "date_claim_amount_agreed": "Date when claim amount was agreed",
            "date_paid_final_amount": "Date when final payment was made",
            "date_fees_paid_final_amount": "Date when final fees were paid",
            "date_reopened": "Date when claim was reopened if applicable",
            "date_claim_denied": "Date when claim was denied",
            "date_claim_withdrawn": "Date when claim was withdrawn",
            "status": "Current status of the claim",
            "refer_to_underwriters": "Flag indicating referral to underwriters",
            "denial_indicator": "Flag indicating if claim was denied",
            "reason_for_denial": "Reason provided for claim denial",
            "claim_total_claimed_amount": "Total amount claimed",
            "settlement_currency_code": "Currency code for settlement",
            "indemnity_amount_paid": "Amount paid as indemnity",
            "fees_amount_paid": "Total fees paid",
            "expenses_paid_amount": "Expenses paid amount",
            "dw_ins_upd_dt": "Data warehouse last update timestamp",
            "org_id": "Organization identifier"
        }
    },
    "dwh.dim_policy": {
        "description": "Stores policy details, effective dates, product information, and geographical coverage.",
        "columns": {
            "Id": "Unique policy record identifier",
            "agreement_id": "Agreement identifier linking policy to contract",
            "policy_number": "Policy number as issued",
            "new_or_renewal": "Indicates if policy is new or renewal",
            "group_reference": "Group reference number",
            "broker_reference": "Broker reference identifier",
            "changed_date": "Date when policy was last changed",
            "effective_date": "Date when policy becomes effective",
            "start_date_time": "Policy start date and time",
            "expiry_date_time": "Policy expiry date and time",
            "renewal_date_time": "Policy renewal date and time",
            "product_code": "Code identifying the insurance product",
            "product_name": "Name of the insurance product",
            "country_code": "ISO country code",
            "country": "Country name where policy is issued",
            "a3_country_code": "ISO 3-letter country code",
            "country_sub_division_code": "Country subdivision code",
            "class_of_business_code": "Business classification code",
            "classof_business_name": "Business classification name",
            "main_line_of_business_name": "Main line of business",
            "insurance_type": "Type of insurance coverage",
            "section_details_number": "Section detail number",
            "section_details_code": "Section detail code",
            "section_details_name": "Section detail name",
            "line_of_business": "Line of business",
            "section_details_description": "Section detail description",
            "dw_ins_upd_dt": "Data warehouse last update timestamp",
            "org_id": "Organization identifier",
            "document_id": "Document identifier"
        }
    },
    "dwh.fact_claims_dtl": {
        "description": "Detailed claim financial information including payments, expenses, fees, and reserves.",
        "columns": {
            "Id": "Unique record identifier",
            "claim_reference_id": "Reference to claim in dim_claims",
            "agreement_id": "Agreement identifier",
            "policy_number": "Associated policy number",
            "org_id": "Organization identifier",
            "riskitems_id": "Risk items identifier",
            "Payment_Detail_Settlement_Currency_Code": "Settlement currency for payments",
            "Paid_Amount": "Total amount paid for this claim detail",
            "Expenses_Paid_Total_Amount": "Total expenses paid",
            "Coverage_Legal_Fees_Total_Paid_Amount": "Legal fees paid under coverage",
            "Defence_Legal_Fees_Total_Paid_Amount": "Defense legal fees paid",
            "Adjusters_Fees_Total_Paid_Amount": "Adjuster fees paid",
            "TPAFees_Paid_Amount": "Third Party Administrator fees paid",
            "Fees_Paid_Amount": "Total fees paid",
            "Incurred_Detail_Settlement_Currency_Code": "Settlement currency for incurred amounts",
            "Indemnity_Amount": "Indemnity amount for this detail",
            "Expenses_Amount": "Expenses amount",
            "Coverage_Legal_Fees_Amount": "Coverage legal fees amount",
            "Defence_Fees_Amount": "Defense fees amount",
            "Adjuster_Fees_Amount": "Adjuster fees amount",
            "TPAFees_Amount": "TPA fees amount",
            "Fees_Amount": "Total fees amount",
            "indemnity_reserves_amount": "Reserved amount for indemnity",
            "dw_ins_upd_dt": "Data warehouse last update timestamp",
            "indemnity_amount_paid": "Actual indemnity amount paid"
        }
    },
    "dwh.fact_premium": {
        "description": "Premium payment transactions including commissions, taxes, brokerage, and currency information.",
        "columns": {
            "Id": "Unique transaction identifier",
            "agreement_id": "Agreement identifier",
            "policy_number": "Associated policy number",
            "org_id": "Organization identifier",
            "riskitems_id": "Risk items identifier",
            "original_currency_code": "Original currency code",
            "total_paid": "Total amount paid in original currency",
            "instalments_amount": "Installment amount",
            "taxes_amount_paid": "Tax amount paid",
            "commission_percentage": "Commission percentage applied",
            "commission_amount_paid": "Commission amount paid",
            "brokerage_amount_paid": "Brokerage fees paid",
            "insurance_amount_paid": "Insurance amount paid",
            "additional_fees_paid": "Additional fees paid",
            "settlement_currency_code": "Settlement currency code",
            "gross_premium_settlement_currency": "Gross premium in settlement currency",
            "brokerage_amount_paid_settlement_currency": "Brokerage amount in settlement currency",
            "net_premium_settlement_currency": "Net premium in settlement currency",
            "commission_amount_paid_settlement_currency": "Commission amount in settlement currency",
            "final_net_premium_settlement_currency": "Final net premium in settlement currency",
            "rate_of_exchange": "Exchange rate applied",
            "total_settlement_amount_paid": "Total settlement amount paid",
            "date_paid": "Date when payment was made",
            "transaction_type": "Type of premium transaction",
            "net_amount": "Net amount",
            "gross_premium_paid_this_time": "Gross premium paid in this transaction",
            "final_net_premium": "Final net premium",
            "tax_amount": "Tax amount",
            "dw_ins_upd_dt": "Data warehouse last update timestamp"
        }
    },
    "dwh.fct_policy": {
        "description": "Policy summary information including premiums, limits, and financial aggregates.",
        "columns": {
            "Id": "Unique policy fact identifier",
            "agreement_id": "Agreement identifier",
            "policy_number": "Policy number",
            "org_id": "Organization identifier",
            "start_date": "Policy start date",
            "annual_premium": "Annual premium amount",
            "sum_insured": "Total sum insured",
            "limit_of_liability": "Liability coverage limit",
            "final_net_premium": "Final net premium after all adjustments",
            "tax_amount": "Tax amount on premium",
            "final_net_premium_settlement_currency": "Final net premium in settlement currency",
            "settlement_currency_code": "Currency code for settlement",
            "gross_premium_before_taxes_amount": "Gross premium before tax calculation",
            "dw_ins_upd_dt": "Data warehouse last update timestamp",
            "document_id": "Document identifier",
            "gross_premium_paid_this_time": "Gross premium paid in current transaction"
        }
    }
}

# Join Conditions and T-SQL Rules
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

# Database Connection Functions (from first file)
def create_session_request_data(
    server: str,
    database: str,
    username: str = None,
    password: str = None,
    driver: str = "ODBC Driver 17 for SQL Server",
    use_trusted_connection: bool = False,
    enabled_tables: list = None,
    disabled_tables: list = None
) -> dict:
    """
    Creates a properly formatted session request for the RAG-Enhanced SQL API.
    Args:
        server (str): SQL Server name or connection string
        database (str): Database name
        username (str, optional): Username for SQL authentication
        password (str, optional): Password for SQL authentication
        driver (str): ODBC driver name
        use_trusted_connection (bool): Use Windows authentication if True
        enabled_tables (list, optional): List of table names to enable (if None, enables all)
        disabled_tables (list, optional): List of table names to disable
    Returns:
        dict: Formatted session request data ready for API call
    """
    # Build credentials section
    credentials = {
        "server": server,
        "database": database,
        "driver": driver
    }
    if use_trusted_connection:
        credentials["trusted_connection"] = True
    else:
        if username and password:
            credentials["username"] = username
            credentials["password"] = password
        else:
            raise ValueError("Username and password required when not using trusted connection")
    
    # Determine which tables to enable
    if enabled_tables is None:
        # Enable all tables by default
        enabled_tables = list(TABLE_DESCRIPTIONS.keys())
    if disabled_tables is None:
        disabled_tables = []
    
    # Build tables configuration using TABLE_DESCRIPTIONS
    tables = {}
    table_descriptions = {}
    column_descriptions = {}
    for table_name, table_info in TABLE_DESCRIPTIONS.items():
        # Enable table if it's in enabled_tables and not in disabled_tables
        is_enabled = table_name in enabled_tables and table_name not in disabled_tables
        tables[table_name] = is_enabled
        table_descriptions[table_name] = table_info["description"]
        column_descriptions[table_name] = table_info["columns"]
    
    return {
        "credentials": credentials,
        "tables": tables,
        "table_descriptions": table_descriptions,
        "column_descriptions": column_descriptions
    }

# Example usage functions
def create_full_session_request(server: str, database: str, username: str, password: str):
    """Create session request with all tables enabled."""
    return create_session_request_data(
        server=server,
        database=database,
        username=username,
        password=password
    )

def create_claims_focused_session_request(server: str, database: str, username: str, password: str):
    """Create session request focused on claims analysis."""
    return create_session_request_data(
        server=server,
        database=database,
        username=username,
        password=password,
        enabled_tables=["dwh.dim_claims", "dwh.fact_claims_dtl", "dwh.dim_policy"]
    )

def create_premium_focused_session_request(server: str, database: str, username: str, password: str):
    """Create session request focused on premium analysis."""
    return create_session_request_data(
        server=server,
        database=database,
        username=username,
        password=password,
        enabled_tables=["dwh.fact_premium", "dwh.fct_policy", "dwh.dim_policy"]
    )

def create_windows_auth_session_request(server: str, database: str, enabled_tables: list = None):
    """Create session request using Windows authentication."""
    return create_session_request_data(
        server=server,
        database=database,
        use_trusted_connection=True,
        enabled_tables=enabled_tables
    )

# --- 2. FastAPI and Pydantic Setup ---
app = FastAPI(title="RAG-Enhanced SQL Query Generator and Analyzer", version="3.0.0")
executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)

class QueryRequest(BaseModel):
    question: str

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

# --- 3. RAG Schema Retrieval System ---
class SchemaRetriever:
    def __init__(self, table_chunks: List[Dict], embedding_model_name: str):
        self.table_chunks = table_chunks
        self.embedding_model_name = embedding_model_name
        self.embeddings_file = os.path.join(EMBEDDINGS_PATH, "table_embeddings.npy")
        self.metadata_file = os.path.join(EMBEDDINGS_PATH, "table_metadata.json")
        # Load or create embeddings
        self.embeddings = None
        self.embedding_model = None
        self.embedding_tokenizer = None
    
    def load_embedding_model(self):
        """Load the sentence transformer model for embeddings."""
        logger.info(f"🔄 Loading embedding model: {self.embedding_model_name}")
        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        logger.info("✅ Embedding model loaded successfully")
    
    def create_embeddings(self):
        """Create embeddings for all table chunks."""
        logger.info("🔄 Creating embeddings for table chunks...")
        if self.embedding_model is None:
            self.load_embedding_model()
        # Prepare texts for embedding
        texts_to_embed = []
        for chunk in self.table_chunks:
            # Combine multiple fields for richer embeddings
            combined_text = f"{chunk['text']} {chunk['description']} {' '.join(chunk['keywords'])}"
            texts_to_embed.append(combined_text)
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts_to_embed)
        # Save embeddings and metadata
        np.save(self.embeddings_file, embeddings)
        with open(self.metadata_file, 'w') as f:
            json.dump({
                "table_names": [chunk["table"] for chunk in self.table_chunks],
                "embedding_dim": embeddings.shape[1],
                "num_tables": len(self.table_chunks)
            }, f)
        self.embeddings = embeddings
        logger.info(f"✅ Created and saved embeddings for {len(self.table_chunks)} tables")
    
    def load_embeddings(self):
        """Load existing embeddings from file."""
        if os.path.exists(self.embeddings_file) and os.path.exists(self.metadata_file):
            self.embeddings = np.load(self.embeddings_file)
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            logger.info(f"✅ Loaded embeddings for {metadata['num_tables']} tables")
            return True
        return False
    
    def retrieve_relevant_tables(self, question: str, top_k: int = 3) -> List[Dict]:
        """Retrieve top-k most relevant tables for the given question."""
        if self.embeddings is None:
            if not self.load_embeddings():
                self.create_embeddings()
        if self.embedding_model is None:
            self.load_embedding_model()
        # Create embedding for the question
        question_embedding = self.embedding_model.encode([question])
        # Calculate cosine similarities
        similarities = cosine_similarity(question_embedding, self.embeddings)[0]
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        # Return relevant table chunks
        relevant_tables = [self.table_chunks[i] for i in top_indices]
        logger.info(f"Retrieved {len(relevant_tables)} relevant tables: {[t['table'] for t in relevant_tables]}")
        return relevant_tables

# --- 4. Enhanced Prompt Engineering ---
def construct_rag_prompt(question: str, relevant_tables: List[Dict]) -> str:
    """Creates a structured prompt using retrieved schema elements."""
    # Build schema section from retrieved tables
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
- Add comments for complex logic if needed
- Ensure all column references are valid according to the provided schema
- Return only the SQL query without explanations
### T-SQL Query
```sql
"""
    return prompt

# --- 5. Device and Model Loading (Enhanced) ---
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
    logger.info(f"Roboto Phi-3-mini analysis model loaded in {time.time() - load_start_time:.2f}s")
    return model, tokenizer

# --- 6. Core Logic Functions (Enhanced) ---
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
            'do_sample': False,
            'num_beams': 1,
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

def generate_sql_with_rag(question: str, retriever: SchemaRetriever) -> Tuple[str, List[str], float]:
    """Generates SQL query using RAG approach with schema retrieval."""
    start_time = time.time()
    # Retrieve relevant tables
    relevant_tables = retriever.retrieve_relevant_tables(question, top_k=3)
    retrieved_table_names = [table['table'] for table in relevant_tables]
    # Construct prompt with retrieved schema
    prompt = construct_rag_prompt(question, relevant_tables)
    # Generate SQL
    response = generate_text_optimized(prompt, app.state.sql_model, app.state.sql_tokenizer, max_new_tokens=300)
    sql_query = extract_sql_from_response(response)
    generation_time = time.time() - start_time
    logger.info(f"SQL generated with RAG in {generation_time:.2f}s using tables: {retrieved_table_names}")
    return sql_query, retrieved_table_names, generation_time

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
   # Initialize RAG retriever
   app.state.retriever = SchemaRetriever(TABLE_CHUNKS, EMBEDDING_MODEL_NAME)
   # Load SQL generation model
   app.state.sql_model, app.state.sql_tokenizer = load_sql_model_and_tokenizer(app.state.device)
   # Load analysis model
   app.state.analysis_model, app.state.analysis_tokenizer = load_analysis_model_and_tokenizer(app.state.device)
   app.state.loop = asyncio.get_event_loop()
   logger.info("🚀 RAG system and both models loaded successfully!")

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
   RAG-enhanced generate -> execute -> analyze pipeline.
   """
   total_start_time = time.time()
   loop = app.state.loop
   try:
       # 1. Generate SQL using RAG-enhanced approach
       sql_query, retrieved_tables, sql_generation_time = await loop.run_in_executor(
           executor, generate_sql_with_rag, request.question, app.state.retriever
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
           "retrieved_tables": retrieved_tables,
           "generated_sql": sql_query,
           "sql_execution_result": sql_execution_result,
           "llm_analysis": llm_analysis,
           "retrieval_time": round(sql_generation_time, 2),  # This includes retrieval time
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
       "rag_retriever_loaded": hasattr(app.state, 'retriever'),
       "device": str(app.state.device) if hasattr(app.state, 'device') else "unknown"
   }

@app.get("/", include_in_schema=False)
async def root():
   """Root endpoint."""
   return {"message": "RAG-Enhanced SQL Generation and Analysis API is running with intelligent schema retrieval."}

# --- 8. Additional Utility Endpoints ---
@app.post("/generate-sql-only")
async def generate_sql_only(request: QueryRequest):
   """Generate SQL query without execution or analysis using RAG."""
   try:
       sql_query, retrieved_tables, generation_time = await app.state.loop.run_in_executor(
           executor, generate_sql_with_rag, request.question, app.state.retriever
       )
       return {
           "question": request.question,
           "retrieved_tables": retrieved_tables,
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
   try:
       relevant_tables = app.state.retriever.retrieve_relevant_tables(request.question, top_k=3)
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



class SessionCredentials(BaseModel):
    server: str
    database: str
    username: str = None
    password: str = None
    driver: str = "ODBC Driver 17 for SQL Server"
    use_trusted_connection: bool = False

# Define explicit column models for each table
class DimClaimsColumns(BaseModel):
    claim_reference_id: bool = False
    date_claim_first_notified: bool = False
    date_of_loss_from: bool = False
    date_claim_opened: bool = False
    date_of_loss_to: bool = False
    cause_of_loss_code: bool = False
    loss_description: bool = False
    date_coverage_confirmed: bool = False
    date_closed: bool = False
    date_claim_amount_agreed: bool = False
    date_paid_final_amount: bool = False
    date_fees_paid_final_amount: bool = False
    date_reopened: bool = False
    date_claim_denied: bool = False
    date_claim_withdrawn: bool = False
    status: bool = False
    refer_to_underwriters: bool = False
    denial_indicator: bool = False
    reason_for_denial: bool = False
    claim_total_claimed_amount: bool = False
    settlement_currency_code: bool = False
    indemnity_amount_paid: bool = False
    fees_amount_paid: bool = False
    expenses_paid_amount: bool = False
    dw_ins_upd_dt: bool = False
    org_id: bool = False

class DimPolicyColumns(BaseModel):
    Id: bool = False
    agreement_id: bool = False
    policy_number: bool = False
    new_or_renewal: bool = False
    group_reference: bool = False
    broker_reference: bool = False
    changed_date: bool = False
    effective_date: bool = False
    start_date_time: bool = False
    expiry_date_time: bool = False
    renewal_date_time: bool = False
    product_code: bool = False
    product_name: bool = False
    country_code: bool = False
    country: bool = False
    a3_country_code: bool = False
    country_sub_division_code: bool = False
    class_of_business_code: bool = False
    classof_business_name: bool = False
    main_line_of_business_name: bool = False
    insurance_type: bool = False
    section_details_number: bool = False
    section_details_code: bool = False
    section_details_name: bool = False
    line_of_business: bool = False
    section_details_description: bool = False
    dw_ins_upd_dt: bool = False
    org_id: bool = False
    document_id: bool = False

class FactClaimsDtlColumns(BaseModel):
    Id: bool = False
    claim_reference_id: bool = False
    agreement_id: bool = False
    policy_number: bool = False
    org_id: bool = False
    riskitems_id: bool = False
    Payment_Detail_Settlement_Currency_Code: bool = False
    Paid_Amount: bool = False
    Expenses_Paid_Total_Amount: bool = False
    Coverage_Legal_Fees_Total_Paid_Amount: bool = False
    Defence_Legal_Fees_Total_Paid_Amount: bool = False
    Adjusters_Fees_Total_Paid_Amount: bool = False
    TPAFees_Paid_Amount: bool = False
    Fees_Paid_Amount: bool = False
    Incurred_Detail_Settlement_Currency_Code: bool = False
    Indemnity_Amount: bool = False
    Expenses_Amount: bool = False
    Coverage_Legal_Fees_Amount: bool = False
    Defence_Fees_Amount: bool = False
    Adjuster_Fees_Amount: bool = False
    TPAFees_Amount: bool = False
    Fees_Amount: bool = False
    indemnity_reserves_amount: bool = False
    dw_ins_upd_dt: bool = False
    indemnity_amount_paid: bool = False

class FactPremiumColumns(BaseModel):
    Id: bool = False
    agreement_id: bool = False
    policy_number: bool = False
    org_id: bool = False
    riskitems_id: bool = False
    original_currency_code: bool = False
    total_paid: bool = False
    instalments_amount: bool = False
    taxes_amount_paid: bool = False
    commission_percentage: bool = False
    commission_amount_paid: bool = False
    brokerage_amount_paid: bool = False
    insurance_amount_paid: bool = False
    additional_fees_paid: bool = False
    settlement_currency_code: bool = False
    gross_premium_settlement_currency: bool = False
    brokerage_amount_paid_settlement_currency: bool = False
    net_premium_settlement_currency: bool = False
    commission_amount_paid_settlement_currency: bool = False
    final_net_premium_settlement_currency: bool = False
    rate_of_exchange: bool = False
    total_settlement_amount_paid: bool = False
    date_paid: bool = False
    transaction_type: bool = False
    net_amount: bool = False
    gross_premium_paid_this_time: bool = False
    final_net_premium: bool = False
    tax_amount: bool = False
    dw_ins_upd_dt: bool = False

class FctPolicyColumns(BaseModel):
    Id: bool = False
    agreement_id: bool = False
    policy_number: bool = False
    org_id: bool = False
    start_date: bool = False
    annual_premium: bool = False
    sum_insured: bool = False
    limit_of_liability: bool = False
    final_net_premium: bool = False
    tax_amount: bool = False
    final_net_premium_settlement_currency: bool = False
    settlement_currency_code: bool = False
    gross_premium_before_taxes_amount: bool = False
    dw_ins_upd_dt: bool = False
    document_id: bool = False
    gross_premium_paid_this_time: bool = False

# Main table selection model
class AvailableTables(BaseModel):
    class Config:
        extra = "forbid"  # This prevents additional properties
    
    dwh_dim_claims: Optional[DimClaimsColumns] = Field(None, alias="dwh.dim_claims")
    dwh_dim_policy: Optional[DimPolicyColumns] = Field(None, alias="dwh.dim_policy")  
    dwh_fact_claims_dtl: Optional[FactClaimsDtlColumns] = Field(None, alias="dwh.fact_claims_dtl")
    dwh_fact_premium: Optional[FactPremiumColumns] = Field(None, alias="dwh.fact_premium")
    dwh_fct_policy: Optional[FctPolicyColumns] = Field(None, alias="dwh.fct_policy")

class CombinedSessionRequest(BaseModel):
    credentials: SessionCredentials
    question: str
    available_tables: AvailableTables = Field(
        description="Select tables and their columns for analysis"
    )

# Helper function to convert AvailableTables to the original format
def convert_available_tables_to_dict(available_tables: AvailableTables) -> Dict[str, Dict[str, bool]]:
    """Convert the structured AvailableTables model back to the original dict format."""
    result = {}
    
    # Map field names to actual table names
    field_to_table_map = {
        'dwh_dim_claims': 'dwh.dim_claims',
        'dwh_dim_policy': 'dwh.dim_policy',
        'dwh_fact_claims_dtl': 'dwh.fact_claims_dtl',
        'dwh_fact_premium': 'dwh.fact_premium',
        'dwh_fct_policy': 'dwh.fct_policy'
    }
    
    for field_name, table_name in field_to_table_map.items():
        table_columns = getattr(available_tables, field_name)
        if table_columns is not None:
            # Convert the Pydantic model to dict
            columns_dict = table_columns.dict()
            result[table_name] = columns_dict
    
    return result

def get_selected_tables_from_input(available_tables_dict: Dict[str, Dict[str, bool]]) -> List[str]:
    """Extract tables that have at least one column selected as True."""
    selected_tables = []
    for table_name, columns in available_tables_dict.items():
        # Check if any column in this table is selected (True)
        if any(column_selected for column_selected in columns.values()):
            selected_tables.append(table_name)
    return selected_tables

def get_selected_columns_for_table(table_name: str, columns: Dict[str, bool]) -> Dict[str, bool]:
    """Get all columns for a table (both selected and unselected) for validation."""
    return columns

def validate_tables_and_columns(available_tables_dict: Dict[str, Dict[str, bool]]) -> Dict[str, Any]:
    """Validate that all provided tables and columns exist in TABLE_DESCRIPTIONS."""
    validation_results = {
        "valid": True,
        "invalid_tables": [],
        "invalid_columns": {},
        "table_column_details": {}
    }
    
    for table_name, columns in available_tables_dict.items():
        # Check if table exists
        if table_name not in TABLE_DESCRIPTIONS:
            validation_results["valid"] = False
            validation_results["invalid_tables"].append(table_name)
            continue
        
        table_info = TABLE_DESCRIPTIONS[table_name]
        valid_columns = set(table_info["columns"].keys())
        input_columns = set(columns.keys())
        invalid_columns = input_columns - valid_columns
        
        # Check if any columns are invalid
        if invalid_columns:
            validation_results["valid"] = False
            validation_results["invalid_columns"][table_name] = list(invalid_columns)
        
        # Build table column details for selected tables only
        if any(columns.values()):  # Only if table has selected columns
            columns_info = {}
            selected_columns_count = 0
            
            for col_name, enabled in columns.items():
                if col_name in table_info["columns"]:  # Only valid columns
                    col_description = table_info["columns"][col_name]
                    columns_info[col_name] = {
                        "description": col_description,
                        "enabled": enabled,
                        "data_type": "varchar/int/datetime"  # You can enhance this
                    }
                    if enabled:
                        selected_columns_count += 1
            
            validation_results["table_column_details"][table_name] = {
                "table_description": table_info["description"],
                "columns": columns_info,
                "total_columns": len(columns_info),
                "selected_columns": selected_columns_count
            }
    
    return validation_results

def perform_rag_comparison(question: str, selected_tables: List[str]) -> Dict[str, Any]:
    """Return success only if all selected tables are within top 3 RAG recommendations."""
    try:
        # Use the actual RAG retriever to get top 3 relevant tables
        relevant_tables = app.state.retriever.retrieve_relevant_tables(question, top_k=3)
        rag_recommended_tables = [table["table"] for table in relevant_tables]
        
        # Check if all selected tables are within RAG recommendations
        tables_not_in_rag = [table for table in selected_tables if table not in rag_recommended_tables]
        
        if not tables_not_in_rag:
            return {
                "validation_status": "success",
                "outofbound": False,
                "message": "All selected tables are within RAG recommendations.",
                "rag_recommended_tables": rag_recommended_tables,
                "selected_tables": selected_tables
            }
        else:
            return {
                "validation_status": "outofbound",
                "outofbound": True,
                "message": f"Tables outside RAG recommendations: {tables_not_in_rag}",
                "rag_recommended_tables": rag_recommended_tables,
                "selected_tables": selected_tables,
                "tables_not_in_rag": tables_not_in_rag
            }
    except Exception as e:
        logger.error(f"RAG comparison failed: {e}")
        # Fallback: allow all selections if RAG fails
        return {
            "validation_status": "rag_unavailable", 
            "outofbound": False,
            "message": f"RAG comparison unavailable: {str(e)}. Proceeding with user selection.",
            "selected_tables": selected_tables
        }

@app.post("/create-session")
async def create_session(request: CombinedSessionRequest):
    """Combined endpoint: Handle credentials, table/column selection, and RAG comparison."""
    try:
        # Convert structured model to original dict format
        available_tables_dict = convert_available_tables_to_dict(request.available_tables)
        
        # Step 1: Extract selected tables (tables with at least one column selected as True)
        selected_tables = get_selected_tables_from_input(available_tables_dict)
        
        if not selected_tables:
            raise HTTPException(status_code=400, detail="At least one table must have selected columns")
        
        # Step 2: Validate tables and columns
        validation_results = validate_tables_and_columns(available_tables_dict)
        
        if not validation_results["valid"]:
            error_details = []
            if validation_results["invalid_tables"]:
                error_details.append(f"Invalid tables: {validation_results['invalid_tables']}")
            if validation_results["invalid_columns"]:
                error_details.append(f"Invalid columns: {validation_results['invalid_columns']}")
            raise HTTPException(status_code=400, detail="; ".join(error_details))
        
        # Step 3: Perform RAG comparison
        rag_comparison = perform_rag_comparison(request.question, selected_tables)
        
        # Step 4: Create session data (you'll need to implement this function)
        # session_data = create_session_request_data(...)
        
        # Step 5: Prepare response
        return {
            "status": "Session created successfully",
            "message": "Session has been created with the selected tables and columns",
            "question_analyzed": request.question,
            "selected_tables": selected_tables,
            "comparison_result": {
                "outofbound": rag_comparison["outofbound"],
                "validation_status": rag_comparison["validation_status"],
                "message": rag_comparison["message"]
            }
        }

        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session creation error: {e}")
        raise HTTPException(status_code=500, detail=f"Session creation failed: {str(e)}")


# @app.post("/create-session-request")
# async def create_session_request(
#     server: str,
#     database: str,
#     username: str = None,
#     password: str = None,
#     driver: str = "ODBC Driver 17 for SQL Server",
#     use_trusted_connection: bool = False,
#     enabled_tables: list = None,
#     disabled_tables: list = None
# ):
#     """Endpoint to create a session request for connecting to SQL database."""
#     try:
#         session_data = create_session_request_data(
#             server=server,
#             database=database,
#             username=username,
#             password=password,
#             driver=driver,
#             use_trusted_connection=use_trusted_connection,
#             enabled_tables=enabled_tables,
#             disabled_tables=disabled_tables
#         )
#         return {
#             "session_request": session_data,
#             "status": "Session request created successfully"
#         }
#     except Exception as e:
#         logger.error(f"Session request creation error: {e}")
#         raise HTTPException(status_code=500, detail=f"Session request creation failed: {str(e)}")

if __name__ == "__main__":
   import uvicorn
   uvicorn.run(app, host="0.0.0.0", port=8000)
