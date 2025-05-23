from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import logging
import time
from functools import lru_cache
import os
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SQL Query Generator with LLM Analysis", version="1.0.0")

# API endpoint configuration
API_ENDPOINT = "http://172.200.64.182:7860/execute"

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

# Pydantic models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    success: bool
    question: str
    generated_sql: str
    sql_execution_result: dict
    llm_analysis: str
    processing_time: float
    files_saved: dict

class SQLGenerator:
    def __init__(self):
        self.sql_model = None
        self.sql_tokenizer = None
        self.analysis_pipeline = None
        self.sql_model_name = "defog/llama-3-sqlcoder-8b"
        self.analysis_model_name = "microsoft/DialoGPT-medium"  # Lighter model for analysis
        self.load_models()
        
        # Cache for tokenized prompts
        self._prompt_cache = {}
    
    def load_models(self):
        """Load both SQL generation and analysis models"""
        try:
            logger.info("Loading SQL generation model...")
            
            # Load SQL tokenizer and model
            self.sql_tokenizer = AutoTokenizer.from_pretrained(self.sql_model_name)
            
            if torch.cuda.is_available():
                self.sql_model = AutoModelForCausalLM.from_pretrained(
                    self.sql_model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    load_in_8bit=True,
                    device_map="auto",
                    use_cache=True,
                    low_cpu_mem_usage=True,
                )
            else:
                self.sql_model = AutoModelForCausalLM.from_pretrained(
                    self.sql_model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    use_cache=True,
                    low_cpu_mem_usage=True,
                )
            
            # Set pad token for SQL model
            if self.sql_tokenizer.pad_token is None:
                self.sql_tokenizer.pad_token = self.sql_tokenizer.eos_token
            
            logger.info("Loading analysis model...")
            
            # Load analysis model using pipeline for easier text generation
            if torch.cuda.is_available():
                self.analysis_pipeline = pipeline(
                    "text-generation",
                    model="microsoft/DialoGPT-medium",
                    tokenizer="microsoft/DialoGPT-medium",
                    device=0,
                    torch_dtype=torch.float16,
                    max_length=512,
                    do_sample=True,
                    temperature=0.7
                )
            else:
                self.analysis_pipeline = pipeline(
                    "text-generation",
                    model="microsoft/DialoGPT-medium",
                    tokenizer="microsoft/DialoGPT-medium",
                    device=-1,
                    max_length=512,
                    do_sample=True,
                    temperature=0.7
                )
            
            # GPU optimizations
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                torch.cuda.empty_cache()
                
            logger.info("Both models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
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
    
    def generate_sql(self, question: str) -> str:
        """Generate SQL query from natural language question"""
        try:
            prompt = self.create_sql_prompt(question)
            
            inputs = self.sql_tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024,
                padding=False
            )
            
            model_device = next(self.sql_model.parameters()).device
            inputs = {k: v.to(model_device, non_blocking=True) for k, v in inputs.items()}
            
            generation_kwargs = {
                "max_new_tokens": 80,
                "temperature": 0.0,
                "do_sample": False,
                "pad_token_id": self.sql_tokenizer.eos_token_id,
                "eos_token_id": self.sql_tokenizer.eos_token_id,
                "use_cache": True,
                "num_beams": 1,
                "early_stopping": True,
            }
            
            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        outputs = self.sql_model.generate(**inputs, **generation_kwargs)
                else:
                    outputs = self.sql_model.generate(**inputs, **generation_kwargs)
            
            generated_text = self.sql_tokenizer.decode(
                outputs[0][len(inputs['input_ids'][0]):], 
                skip_special_tokens=True
            )
            
            sql_query = self.extract_sql_from_response(generated_text)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return sql_query
            
        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.error(f"Error generating SQL: {str(e)}")
            return ""
    
    def analyze_results(self, question: str, sql_query: str, sql_result: dict) -> str:
        """Use LLM to analyze SQL results and provide insights"""
        try:
            # Create analysis prompt
            analysis_prompt = f"""Question: {question}
SQL Query: {sql_query}
Results: {json.dumps(sql_result, indent=2, default=str)[:1000]}

Please provide a clear analysis of these results, including:
1. What the data shows
2. Key insights
3. Summary of findings

Analysis:"""
            
            # Generate analysis using the pipeline
            response = self.analysis_pipeline(
                analysis_prompt,
                max_new_tokens=200,
                num_return_sequences=1,
                pad_token_id=self.analysis_pipeline.tokenizer.eos_token_id
            )
            
            if response and len(response) > 0:
                generated_text = response[0]['generated_text']
                # Extract only the new text after the prompt
                analysis = generated_text[len(analysis_prompt):].strip()
                return analysis if analysis else "Analysis could not be generated."
            else:
                return "Analysis could not be generated."
                
        except Exception as e:
            logger.error(f"Error analyzing results: {str(e)}")
            return f"Error generating analysis: {str(e)}"

def execute_sql_via_api(sql_query: str):
    """Execute SQL query by calling the API endpoint"""
    try:
        payload = {"query": sql_query}
        response = requests.post(
            API_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "success": False,
                "error": f"API request failed with status {response.status_code}: {response.text}"
            }
            
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"API connection error: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

def save_to_files(sql_query: str, sql_result: dict, llm_analysis: str, question: str) -> dict:
    """Save SQL and analysis results to files"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save SQL result
        sql_filename = f"sql_{timestamp}.txt"
        with open(sql_filename, 'w', encoding='utf-8') as f:
            f.write(f"Question: {question}\n")
            f.write(f"Generated SQL Query:\n{sql_query}\n\n")
            f.write("SQL Execution Result:\n")
            f.write(json.dumps(sql_result, indent=2, default=str))
        
        # Save LLM analysis
        response_filename = f"response_{timestamp}.txt"
        with open(response_filename, 'w', encoding='utf-8') as f:
            f.write(f"Question: {question}\n\n")
            f.write(f"SQL Query:\n{sql_query}\n\n")
            f.write("LLM Analysis:\n")
            f.write(llm_analysis)
        
        return {
            "sql_file": sql_filename,
            "response_file": response_filename,
            "saved_successfully": True
        }
        
    except Exception as e:
        logger.error(f"Error saving files: {str(e)}")
        return {
            "sql_file": None,
            "response_file": None,
            "saved_successfully": False,
            "error": str(e)
        }

# Initialize the SQL generator globally
sql_generator = None

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global sql_generator
    logger.info("Initializing SQL Generator...")
    try:
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
    """Process natural language question and return SQL + analysis"""
    if not sql_generator:
        raise HTTPException(status_code=500, detail="SQL generator not initialized")
    
    start_time = time.time()
    
    try:
        # Generate SQL query
        logger.info(f"Processing question: {request.question}")
        sql_query = sql_generator.generate_sql(request.question)
        
        if not sql_query:
            raise HTTPException(status_code=400, detail="Failed to generate SQL query")
        
        # Execute SQL query
        sql_result = execute_sql_via_api(sql_query)
        
        # Generate LLM analysis
        llm_analysis = sql_generator.analyze_results(request.question, sql_query, sql_result)
        
        # Save to files
        files_info = save_to_files(sql_query, sql_result, llm_analysis, request.question)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        return QueryResponse(
            success=True,
            question=request.question,
            generated_sql=sql_query,
            sql_execution_result=sql_result,
            llm_analysis=llm_analysis,
            processing_time=processing_time,
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
        "api_endpoint": API_ENDPOINT
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
