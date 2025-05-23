import requests
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
import re
import time
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

JOIN_RULES = """
Based on the user's question, generate an SQL query. The query should always include
necessary table joins based on the following rules:
- For a policy-related question, join 'fct_policy' and 'dim_policy' on 'policy_number' and 'org_id'.
- For a claims-related question, join 'fact_claims_dtl' and 'dim_claims' on 'claim_reference_id' and 'org_id'.
- Always use proper T-SQL syntax.
- Return only the SQL query without any explanatory text.
"""

class SQLGenerator:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = "defog/llama-3-sqlcoder-8b"
        self.load_model()
        
        # Cache for tokenized prompts
        self._prompt_cache = {}
    
    def load_model(self):
        """Load the SQLCoder model and tokenizer with maximum GPU optimization"""
        try:
            logger.info("Loading Defog SQLCoder model...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            if torch.cuda.is_available():
                logger.info("Loading model with aggressive GPU optimization...")
                
                # Use the most aggressive GPU optimizations
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,  # Use bfloat16 for better performance than float16
                    load_in_4bit=True,  # More aggressive quantization than 8bit
                    device_map="auto",
                    use_cache=True,
                    low_cpu_mem_usage=True,
                    # Additional optimization parameters
                    attn_implementation="flash_attention_2" if hasattr(torch.nn, 'attention') else None,
                )
                
                # Compile model for faster inference (PyTorch 2.0+)
                if hasattr(torch, 'compile'):
                    logger.info("Compiling model for faster inference...")
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                
            else:
                logger.info("Loading model for CPU with optimizations...")
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
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.cuda.empty_cache()
                
                # Warm up the model with a dummy forward pass
                self._warmup_model()
                
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
                
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise e
    
    def _warmup_model(self):
        """Warm up the model to optimize first inference"""
        try:
            logger.info("Warming up model...")
            dummy_prompt = "SELECT COUNT(*) FROM table;"
            inputs = self.tokenizer(dummy_prompt, return_tensors="pt", max_length=100)
            model_device = next(self.model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            
            with torch.no_grad():
                _ = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)
            
            torch.cuda.empty_cache()
            logger.info("Model warmup completed")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    @lru_cache(maxsize=128)
    def create_prompt(self, question: str) -> str:
        """Create a cached, optimized prompt"""
        # Shorter, more focused prompt to reduce token count
        prompt = f"""### Task
Generate T-SQL for: {question}

### Schema
{DATABASE_SCHEMA}

### SQL
```sql"""
        return prompt
    
    def extract_sql_from_response(self, response: str) -> str:
        """Optimized SQL extraction with early termination"""
        response = response.strip()
        if not response:
            return ""
        
        # Fast path: Look for SQL in code blocks
        if "```sql" in response:
            start = response.find("```sql") + 6
            end = response.find("```", start)
            if end != -1:
                return self.clean_sql_query(response[start:end].strip())
        
        # Fast path: Find SELECT and stop at semicolon
        if "SELECT" in response.upper():
            select_pos = response.upper().find("SELECT")
            after_select = response[select_pos:]
            semicolon_pos = after_select.find(';')
            if semicolon_pos != -1:
                sql = after_select[:semicolon_pos + 1]
                return self.clean_sql_query(sql)
        
        return ""
    
    def clean_sql_query(self, sql_query: str) -> str:
        """Simplified SQL cleaning"""
        lines = [line.strip() for line in sql_query.split('\n') if line.strip()]
        if not lines:
            return ""
        
        # Keep only SQL-looking lines
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
        """Highly optimized SQL generation"""
        start_time = time.time()
        
        try:
            # Create optimized prompt
            prompt = self.create_prompt(question)
            
            # Optimized tokenization with reduced max length
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024,  # Reduced from 2048
                padding=False
            )
            
            # Move to device efficiently
            model_device = next(self.model.parameters()).device
            inputs = {k: v.to(model_device, non_blocking=True) for k, v in inputs.items()}
            
            # Highly optimized generation parameters
            generation_kwargs = {
                "max_new_tokens": 80,     # Reduced from 150
                "temperature": 0.0,       # Deterministic
                "do_sample": False,       # No sampling
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
                "num_beams": 1,          # No beam search
                "early_stopping": True,
                "repetition_penalty": 1.0,
            }
            
            # Generate with optimizations
            with torch.no_grad():
                if torch.cuda.is_available():
                    # Use autocast for faster inference
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        outputs = self.model.generate(**inputs, **generation_kwargs)
                else:
                    outputs = self.model.generate(**inputs, **generation_kwargs)
            
            # Efficient decoding
            generated_text = self.tokenizer.decode(
                outputs[0][len(inputs['input_ids'][0]):], 
                skip_special_tokens=True
            )
            
            # Extract SQL
            sql_query = self.extract_sql_from_response(generated_text)
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            end_time = time.time()
            logger.info(f"SQL generation took {end_time - start_time:.2f} seconds")
            
            return sql_query
            
        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.error(f"Error generating SQL: {str(e)}")
            return ""

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

def process_question(sql_generator, question: str):
    """Process a natural language question with timing"""
    print(f"\nQuestion: {question}")
    print("-" * 50)
    
    total_start = time.time()
    
    # Generate SQL
    print("Generating SQL query...")
    sql_start = time.time()
    sql_query = sql_generator.generate_sql(question)
    sql_end = time.time()
    
    if not sql_query:
        print("Failed to generate SQL query")
        return
    
    print(f"Generated SQL (took {sql_end - sql_start:.2f}s):\n{sql_query}")
    print("-" * 30)
    
    # Execute via API
    print("Executing SQL via API...")
    api_start = time.time()
    result = execute_sql_via_api(sql_query)
    api_end = time.time()
    
    total_end = time.time()
    
    print(f"API execution took {api_end - api_start:.2f}s")
    print(f"Total processing time: {total_end - total_start:.2f}s")
    
    if result.get("success"):
        print("Query executed successfully!")
        
        if result.get("data") is not None:
            data = result.get("data", [])
            columns = result.get("columns", [])
            row_count = result.get("row_count", 0)
            
            print(f"Rows returned: {row_count}")
            print(f"Columns: {columns}")
            
            if data:
                print("\nQuery Results:")
                print("=" * 40)
                
                if columns:
                    header = " | ".join(str(col) for col in columns)
                    print(header)
                    print("-" * len(header))
                
                for i, row in enumerate(data[:5]):  # Show only first 5 rows
                    if isinstance(row, dict):
                        row_values = [str(row.get(col, '')) for col in columns] if columns else [str(v) for v in row.values()]
                        print(" | ".join(row_values))
                    else:
                        print(str(row))
                    
                if len(data) > 5:
                    print(f"... and {len(data) - 5} more rows")
            else:
                print("No data returned (empty result set)")
                
        else:
            rows_affected = result.get("row_count", 0)
            print(f"Query executed successfully. Rows affected: {rows_affected}")
    else:
        print(f"Query execution failed: {result.get('error', 'Unknown error')}")
    
    print("=" * 60)

def main():
    """Main function with optimized initialization"""
    print("Initializing optimized SQL Generator...")
    init_start = time.time()
    
    try:
        sql_generator = SQLGenerator()
        init_end = time.time()
        print(f"SQL Generator ready! (Initialization took {init_end - init_start:.2f}s)")
    except Exception as e:
        print(f"Failed to initialize SQL generator: {e}")
        return
    
    print(f"\nAPI Endpoint: {API_ENDPOINT}")
    print("=" * 60)
    print("Optimized Natural Language to SQL Query Generator")
    print("=" * 60)
    print("Enter your questions (type 'quit' to exit):")
    
    while True:
        try:
            question = input("\nYour question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            process_question(sql_generator, question)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
