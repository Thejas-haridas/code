from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
import time
from functools import lru_cache
import os
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="SQL Query Generator with LLM Analysis", version="1.0.0")

# API endpoint configuration
API_ENDPOINT = "http://172.200.64.182:7860/execute"

# File path configuration
SAVE_PATH = "/home/text_sql"
SINGLE_FILE_PATH = os.path.join(SAVE_PATH, "query_results.txt")

# Ensure directory exists
os.makedirs(SAVE_PATH, exist_ok=True)

# Check GPU availability and set device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")
    print("No GPU detected, using CPU")

# Database schema
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

# Enhanced Pydantic models with timing information
class TimingInfo(BaseModel):
    sql_generation_time: float
    sql_execution_time: float
    llm_analysis_time: float
    file_save_time: float
    total_processing_time: float

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    success: bool
    question: str
    generated_sql: str
    sql_execution_result: dict
    llm_analysis: str
    processing_time: float
    timing_breakdown: TimingInfo
    files_saved: dict

class TimingContext:
    """Context manager for timing operations"""
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"TIMING: Starting {self.operation_name}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.info(f"TIMING: {self.operation_name} completed in {duration:.3f} seconds")
        
    def get_duration(self) -> float:
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0

class SQLGenerator:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = "defog/llama-3-sqlcoder-8b"
        self.load_model()
        
        # Cache for tokenized prompts
        self._prompt_cache = {}
    
    def load_model(self):
        """Load the LLM model for both SQL generation and analysis"""
        try:
            logger.info("Loading LLM model...")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            if torch.cuda.is_available():
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    load_in_8bit=True,
                    device_map="auto",
                    use_cache=True,
                    low_cpu_mem_usage=True,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    use_cache=True,
                    low_cpu_mem_usage=True,
                )
            
            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # GPU optimizations
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                torch.cuda.empty_cache()
                
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise e
    
    @lru_cache(maxsize=128)
    def create_sql_prompt(self, question: str) -> str:
        """Create a cached, optimized prompt for SQL generation"""
        prompt = f"""### Task
Generate T-SQL for: {question}

### Schema
{DATABASE_SCHEMA}

### SQL
```sql"""
        return prompt
    
    def create_analysis_prompt(self, question: str, sql_query: str, sql_result: dict) -> str:
        """Create prompt for analysis generation"""
        prompt = f"""### Task
Analyze the following SQL query results and provide insights.

### Question
{question}

### SQL Query
{sql_query}

### Results
{json.dumps(sql_result, indent=2, default=str)[:1000]}

### Analysis
Please provide a clear analysis including:
1. What the data shows
2. Key insights
3. Summary of findings

Analysis:"""
        return prompt
    
    def extract_sql_from_response(self, response: str) -> str:
        """Extract SQL from model response"""
        response = response.strip()
        if not response:
            return ""
        
        if "```sql" in response:
            start = response.find("```sql") + 6
            end = response.find("```", start)
            if end != -1:
                return self.clean_sql_query(response[start:end].strip())
        
        if "SELECT" in response.upper():
            select_pos = response.upper().find("SELECT")
            after_select = response[select_pos:]
            semicolon_pos = after_select.find(';')
            if semicolon_pos != -1:
                sql = after_select[:semicolon_pos + 1]
                return self.clean_sql_query(sql)
        
        return ""
    
    def extract_analysis_from_response(self, response: str, prompt: str) -> str:
        """Extract analysis from model response"""
        response = response.strip()
        if not response:
            return "Analysis could not be generated."
        
        # Remove the original prompt from the response
        if prompt in response:
            analysis = response.replace(prompt, "").strip()
        else:
            analysis = response
        
        # Clean up the analysis
        if analysis.startswith("Analysis:"):
            analysis = analysis[9:].strip()
        
        return analysis if analysis else "Analysis could not be generated."
    
    def clean_sql_query(self, sql_query: str) -> str:
        """Clean and format SQL query"""
        lines = [line.strip() for line in sql_query.split('\n') if line.strip()]
        if not lines:
            return ""
        
        sql_lines = []
        for line in lines:
            if (line.upper().startswith(('SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 
                                      'GROUP', 'ORDER', 'HAVING', 'UNION', 'WITH')) or
                any(keyword in line.upper() for keyword in [' AS ', ' ON ', ' AND ', ' OR '])):
                sql_lines.append(line)
        
        if not sql_lines:
            return sql_query.strip()
        
        result = '\n'.join(sql_lines).strip()
        return result + ';' if result and not result.endswith(';') else result
    
    def generate_text(self, prompt: str, max_new_tokens: int = 80, temperature: float = 0.0) -> str:
        """Generate text using the LLM model with detailed timing"""
        try:
            # Tokenization timing
            tokenize_start = time.time()
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048,
                padding=False
            )
            tokenize_time = time.time() - tokenize_start
            logger.info(f"TIMING: Tokenization took {tokenize_time:.3f} seconds")
            
            # Device transfer timing
            transfer_start = time.time()
            model_device = next(self.model.parameters()).device
            inputs = {k: v.to(model_device, non_blocking=True) for k, v in inputs.items()}
            transfer_time = time.time() - transfer_start
            logger.info(f"TIMING: Device transfer took {transfer_time:.3f} seconds")
            
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
                "num_beams": 1 if temperature == 0 else None,
                "early_stopping": True,
            }
            
            # Model inference timing
            inference_start = time.time()
            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        outputs = self.model.generate(**inputs, **generation_kwargs)
                else:
                    outputs = self.model.generate(**inputs, **generation_kwargs)
            inference_time = time.time() - inference_start
            logger.info(f"TIMING: Model inference took {inference_time:.3f} seconds")
            
            # Decoding timing
            decode_start = time.time()
            generated_text = self.tokenizer.decode(
                outputs[0][len(inputs['input_ids'][0]):], 
                skip_special_tokens=True
            )
            decode_time = time.time() - decode_start
            logger.info(f"TIMING: Decoding took {decode_time:.3f} seconds")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            total_llm_time = tokenize_time + transfer_time + inference_time + decode_time
            logger.info(f"TIMING: Total LLM generation time: {total_llm_time:.3f} seconds")
            
            return generated_text
            
        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.error(f"Error generating text: {str(e)}")
            return ""
    
    def generate_sql(self, question: str) -> str:
        """Generate SQL query from natural language question"""
        with TimingContext("SQL Generation"):
            prompt = self.create_sql_prompt(question)
            generated_text = self.generate_text(prompt, max_new_tokens=80, temperature=0.0)
            return self.extract_sql_from_response(generated_text)
    
    def analyze_results(self, question: str, sql_query: str, sql_result: dict) -> str:
        """Use LLM to analyze SQL results and provide insights"""
        try:
            with TimingContext("LLM Analysis"):
                prompt = self.create_analysis_prompt(question, sql_query, sql_result)
                generated_text = self.generate_text(prompt, max_new_tokens=300, temperature=0.7)
                return self.extract_analysis_from_response(generated_text, prompt)
                
        except Exception as e:
            logger.error(f"Error analyzing results: {str(e)}")
            return f"Error generating analysis: {str(e)}"

def execute_sql_via_api(sql_query: str):
    """Execute SQL query by calling the API endpoint with timing"""
    try:
        with TimingContext("SQL API Execution") as timer:
            payload = {"query": sql_query}
            
            # Network request timing
            request_start = time.time()
            response = requests.post(
                API_ENDPOINT,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            request_time = time.time() - request_start
            logger.info(f"TIMING: HTTP request took {request_time:.3f} seconds")
            
            if response.status_code == 200:
                # JSON parsing timing
                parse_start = time.time()
                result = response.json()
                parse_time = time.time() - parse_start
                logger.info(f"TIMING: JSON parsing took {parse_time:.3f} seconds")
                return result
            else:
                return {
                    "success": False,
                    "error": f"API request failed with status {response.status_code}: {response.text}"
                }
            
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"API connection error: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

def append_to_single_file(sql_query: str, sql_result: dict, llm_analysis: str, question: str) -> dict:
    """Append query results to the single file with timing"""
    try:
        with TimingContext("File Save Operation") as timer:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Prepare content to append
            content = f"""
{'='*80}
QUERY EXECUTED AT: {timestamp}
{'='*80}

QUESTION: {question}

GENERATED SQL QUERY:
{sql_query}

SQL EXECUTION RESULT:
{json.dumps(sql_result, indent=2, default=str)}

LLM ANALYSIS:
{llm_analysis}

{'='*80}

"""
            
            # File write timing
            write_start = time.time()
            with open(SINGLE_FILE_PATH, 'a', encoding='utf-8') as f:
                f.write(content)
            write_time = time.time() - write_start
            logger.info(f"TIMING: File write took {write_time:.3f} seconds")
        
        return {
            "file_path": SINGLE_FILE_PATH,
            "saved_successfully": True,
            "timestamp": timestamp
        }
        
    except Exception as e:
        logger.error(f"Error saving to file: {str(e)}")
        return {
            "file_path": None,
            "saved_successfully": False,
            "error": str(e)
        }

def clear_single_file():
    """Clear the single file content (called on API initialization)"""
    try:
        with open(SINGLE_FILE_PATH, 'w', encoding='utf-8') as f:
            f.write(f"SQL Query Results Log - Initialized at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
        logger.info(f"Cleared and initialized file: {SINGLE_FILE_PATH}")
        return True
    except Exception as e:
        logger.error(f"Error clearing file: {str(e)}")
        return False

# Initialize the SQL generator globally
sql_generator = None

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global sql_generator
    logger.info("Initializing SQL Generator...")
    try:
        # Clear the single file on startup
        clear_single_file()
        
        # Initialize the SQL generator
        sql_generator = SQLGenerator()
        logger.info("SQL Generator initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize SQL generator: {e}")
        raise e

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "SQL Query Generator with LLM Analysis API is running!"}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process natural language question and return SQL + analysis with detailed timing"""
    if not sql_generator:
        raise HTTPException(status_code=500, detail="SQL generator not initialized")
    
    overall_start_time = time.time()
    
    try:
        logger.info(f"TIMING: Starting query processing for: {request.question}")
        
        # Generate SQL query with timing
        sql_start = time.time()
        sql_query = sql_generator.generate_sql(request.question)
        sql_generation_time = time.time() - sql_start
        
        if not sql_query:
            raise HTTPException(status_code=400, detail="Failed to generate SQL query")
        
        # Execute SQL query with timing
        execution_start = time.time()
        sql_result = execute_sql_via_api(sql_query)
        sql_execution_time = time.time() - execution_start
        
        # Generate LLM analysis with timing
        analysis_start = time.time()
        llm_analysis = sql_generator.analyze_results(request.question, sql_query, sql_result)
        llm_analysis_time = time.time() - analysis_start
        
        # Save to single file with timing
        save_start = time.time()
        files_info = append_to_single_file(sql_query, sql_result, llm_analysis, request.question)
        file_save_time = time.time() - save_start
        
        overall_end_time = time.time()
        total_processing_time = overall_end_time - overall_start_time
        
        # Create timing breakdown
        timing_info = TimingInfo(
            sql_generation_time=sql_generation_time,
            sql_execution_time=sql_execution_time,
            llm_analysis_time=llm_analysis_time,
            file_save_time=file_save_time,
            total_processing_time=total_processing_time
        )
        
        # Log comprehensive timing summary
        logger.info(f"TIMING SUMMARY:")
        logger.info(f"  SQL Generation: {sql_generation_time:.3f}s ({sql_generation_time/total_processing_time*100:.1f}%)")
        logger.info(f"  SQL Execution: {sql_execution_time:.3f}s ({sql_execution_time/total_processing_time*100:.1f}%)")
        logger.info(f"  LLM Analysis: {llm_analysis_time:.3f}s ({llm_analysis_time/total_processing_time*100:.1f}%)")
        logger.info(f"  File Save: {file_save_time:.3f}s ({file_save_time/total_processing_time*100:.1f}%)")
        logger.info(f"  TOTAL: {total_processing_time:.3f}s")
        
        return QueryResponse(
            success=True,
            question=request.question,
            generated_sql=sql_query,
            sql_execution_result=sql_result,
            llm_analysis=llm_analysis,
            processing_time=total_processing_time,
            timing_breakdown=timing_info,
            files_saved=files_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "sql_generator_ready": sql_generator is not None,
        "gpu_available": torch.cuda.is_available(),
        "device_info": str(device),
        "api_endpoint": API_ENDPOINT,
        "save_path": SAVE_PATH,
        "single_file_path": SINGLE_FILE_PATH
    }

@app.get("/timing-test")
async def timing_test():
    """Test endpoint to measure component timing without processing a real query"""
    if not sql_generator:
        raise HTTPException(status_code=500, detail="SQL generator not initialized")
    
    start_time = time.time()
    
    # Test SQL generation
    test_question = "How many policies are there"
    sql_start = time.time()
    test_sql = sql_generator.generate_sql(test_question)
    sql_time = time.time() - sql_start
    
    # Test API call with a simple query
    api_start = time.time()
    api_result = execute_sql_via_api("SELECT 1 as test")
    api_time = time.time() - api_start
    
    total_time = time.time() - start_time
    
    return {
        "sql_generation_time": sql_time,
        "api_call_time": api_time,
        "total_test_time": total_time,
        "generated_sql": test_sql,
        "api_result": api_result
    }

@app.post("/clear-log")
async def clear_log():
    """Manually clear the log file"""
    success = clear_single_file()
    if success:
        return {"message": "Log file cleared successfully", "file_path": SINGLE_FILE_PATH}
    else:
        raise HTTPException(status_code=500, detail="Failed to clear log file")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
