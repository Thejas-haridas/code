import requests
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
import re

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
    torch.cuda.empty_cache()  # Clear GPU cache
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
        self.load_model()
    
    def load_model(self):
        """Load the SQLCoder model and tokenizer with GPU optimization"""
        try:
            logger.info("Loading Defog SQLCoder model...")
            model_name = "defog/llama-3-sqlcoder-8b"
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Set optimal settings based on available hardware
            if torch.cuda.is_available():
                logger.info("Loading model with GPU optimization...")
                # For GPU with sufficient memory
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,  # Use FP16 for GPU
                    load_in_8bit=True,
                    device_map="auto",
                    use_cache=True,
                    low_cpu_mem_usage=True,
                )
            else:
                logger.info("Loading model for CPU...")
                # For CPU fallback
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,  # Use FP32 for CPU
                    device_map="cpu",
                    use_cache=True,
                    low_cpu_mem_usage=True,
                )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Enable optimizations
            if torch.cuda.is_available():
                # Enable GPU optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                
                # Clear GPU cache
                torch.cuda.empty_cache()
                
                # Get memory info
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
                
            logger.info("Model loaded successfully!")
            logger.info(f"Model device: {next(self.model.parameters()).device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise e
    
    def create_prompt(self, question: str) -> str:
        """Create a properly formatted prompt for the SQLCoder model"""
        
        prompt = f"""### Task
Generate a T-SQL query to answer this question: `{question}`. Return only the SQL query, ending with a semicolon. Do not include any explanations, comments, or additional text like 'assistant:'.

### Database Schema
{DATABASE_SCHEMA}

### Instructions
{JOIN_RULES}

### SQL Query
```sql"""
        
        return prompt
    
    def extract_sql_from_response(self, response: str) -> str:
        """Extract SQL query from model response - return only clean SQL"""
        
        # Clean up the response first
        response = response.strip()
        
        if not response:
            return ""
        
        # Method 1: Look for SQL code blocks first
        if "```sql" in response:
            sql_start = response.find("```sql") + 6
            sql_end = response.find("```", sql_start)
            if sql_end != -1:
                sql_query = response[sql_start:sql_end].strip()
                return self.clean_sql_query(sql_query)
        
        # Method 2: Find first semicolon and extract everything before it as SQL
        semicolon_pos = response.find(';')
        if semicolon_pos != -1:
            # Extract everything up to and including the first semicolon
            potential_sql = response[:semicolon_pos + 1].strip()
            
            # Check if it contains SQL keywords
            if any(keyword in potential_sql.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH']):
                return self.clean_sql_query(potential_sql)
        
        # Method 3: Look for SELECT statements and stop at unwanted content
        if "SELECT" in response.upper():
            # Split by common stop words that indicate conversational text
            stop_patterns = [
                'assistant', 'i\'m happy', 'however,', 'it seems', 'could you',
                'rephrase', 'provide more', 'better understand', 'i can better',
                'note:', 'explanation:', 'here is', 'here\'s', 'let me', 'would you',
                'please', 'sorry', 'apologize', 'understand', 'help you'
            ]
            
            # Find the earliest occurrence of any stop pattern
            earliest_stop = len(response)
            for pattern in stop_patterns:
                pos = response.lower().find(pattern.lower())
                if pos != -1 and pos < earliest_stop:
                    earliest_stop = pos
            
            # Extract text before the stop pattern
            if earliest_stop < len(response):
                response = response[:earliest_stop].strip()
            
            # Now extract SQL
            lines = response.split('\n')
            sql_lines = []
            capturing = False
            
            for line in lines:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('--'):
                    continue
                
                # Start capturing when we see SELECT, WITH, INSERT, UPDATE, DELETE
                if any(keyword in line.upper() for keyword in ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE']):
                    capturing = True
                
                if capturing:
                    # Stop if we hit conversational text
                    if any(phrase in line.lower() for phrase in stop_patterns):
                        break
                    
                    sql_lines.append(line)
                    
                    # Stop at semicolon
                    if line.endswith(';'):
                        break
            
            if sql_lines:
                sql_result = '\n'.join(sql_lines).strip()
                return self.clean_sql_query(sql_result)
        
        # Method 4: Use regex to extract SQL patterns - stop at first semicolon
        sql_pattern = r'((?:SELECT|WITH|INSERT|UPDATE|DELETE).*?;)'
        matches = re.findall(sql_pattern, response, re.IGNORECASE | re.DOTALL)
        if matches:
            sql_query = matches[0].strip()
            return self.clean_sql_query(sql_query)
        
        # Return empty string if no SQL found
        return ""
    
    def clean_sql_query(self, sql_query: str) -> str:
        """Clean and validate the extracted SQL query - return only SQL"""
        
        # Remove any remaining conversational text
        lines = sql_query.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip lines that look like conversational text or are empty
            if not line:
                continue
                
            # Skip obvious conversational lines
            if any(phrase in line.lower() for phrase in [
                'assistant:', 'i\'m happy', 'however,', 'it seems', 'could you',
                'rephrase', 'provide more', 'better understand', 'note:', 'explanation:',
                'here is', 'here\'s', 'let me', 'would you', 'please', 'sorry',
                'apologize', 'understand', 'help you', 'question', 'context'
            ]):
                break
            
            # Only add lines that look like SQL
            if (line.upper().startswith(('SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 
                                      'GROUP', 'ORDER', 'HAVING', 'UNION', 'WITH', 'INSERT', 
                                      'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP')) or
                line.strip().endswith((',', ';')) or
                any(keyword in line.upper() for keyword in [' AS ', ' ON ', ' AND ', ' OR ', ' IN ', ' NOT '])):
                clean_lines.append(line)
            elif clean_lines:  # If we've started collecting SQL and hit non-SQL, stop
                break
        
        if not clean_lines:
            return ""
        
        cleaned_sql = '\n'.join(clean_lines).strip()
        
        # Remove leading semicolons
        while cleaned_sql.startswith(';'):
            cleaned_sql = cleaned_sql[1:].strip()
        
        # Ensure it ends with semicolon if it doesn't already
        if cleaned_sql and not cleaned_sql.endswith(';'):
            cleaned_sql += ';'
        
        return cleaned_sql
    
    def generate_sql(self, question: str, max_tokens: int = 150, temperature: float = 0.0) -> str:
        """Generate T-SQL query from natural language question - return only SQL"""
        
        try:
            # Create prompt
            prompt = self.create_prompt(question)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048,
                padding=False
            )
            
            # Move inputs to the same device as model
            model_device = next(self.model.parameters()).device
            inputs = {k: v.to(model_device, non_blocking=True) for k, v in inputs.items()}
            
            # Generation settings - very restrictive to get only SQL
            generation_kwargs = {
                "max_new_tokens": max_tokens,  # Very limited tokens
                "temperature": temperature,    # No randomness
                "do_sample": False,           # Deterministic output
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": 1.0,   # No penalty to avoid weird behavior
                "early_stopping": True,
                "use_cache": True,
                "num_beams": 1,
            }
            
            # Generate response
            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.amp.autocast('cuda'):
                        outputs = self.model.generate(**inputs, **generation_kwargs)
                else:
                    outputs = self.model.generate(**inputs, **generation_kwargs)
            
            # Move output back to CPU for decoding if needed
            if torch.cuda.is_available():
                outputs = outputs.cpu()
            
            # Decode response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the new content (remove the prompt part)
            new_content = generated_text[len(prompt):].strip()
            
            # Extract SQL from the generated text
            sql_query = self.extract_sql_from_response(new_content)
            
            # Clear GPU cache after generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return sql_query if sql_query else ""
            
        except Exception as e:
            # Clear GPU cache on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.error(f"Error generating SQL: {str(e)}")
            return ""

def execute_sql_via_api(sql_query: str):
    """Execute SQL query by calling the API endpoint"""
    try:
        # Prepare the request payload
        payload = {
            "query": sql_query
        }
        
        # Make API call
        response = requests.post(
            API_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            return {
                "success": False,
                "error": f"API request failed with status {response.status_code}: {response.text}"
            }
            
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"API connection error: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }

def process_question(sql_generator, question: str):
    """Process a natural language question: generate SQL and execute it via API"""
    print(f"\nQuestion: {question}")
    print("-" * 50)
    
    # Generate SQL
    print("Generating SQL query...")
    sql_query = sql_generator.generate_sql(question)
    
    if not sql_query:
        print("❌ Failed to generate SQL query")
        return
    
    print(f"Generated SQL:\n{sql_query}")
    print("-" * 30)
    
    # Execute via API
    print("Executing SQL via API...")
    result = execute_sql_via_api(sql_query)
    
    if result.get("success"):
        print("✅ Query executed successfully!")
        
        # Check if we have data (SELECT query result)
        if result.get("data") is not None:
            data = result.get("data", [])
            columns = result.get("columns", [])
            row_count = result.get("row_count", 0)
            
            print(f"Rows returned: {row_count}")
            print(f"Columns: {columns}")
            
            if data:
                print("\nQuery Results:")
                print("=" * 40)
                
                # Print column headers
                if columns:
                    header = " | ".join(str(col) for col in columns)
                    print(header)
                    print("-" * len(header))
                
                # Print data rows
                for i, row in enumerate(data):
                    if i < 10:  # Show first 10 rows
                        if isinstance(row, dict):
                            row_values = [str(row.get(col, '')) for col in columns] if columns else [str(v) for v in row.values()]
                            print(" | ".join(row_values))
                        else:
                            print(str(row))
                    
                if len(data) > 10:
                    print(f"... and {len(data) - 10} more rows")
            else:
                print("No data returned (empty result set)")
                
        else:
            # Non-SELECT query (INSERT, UPDATE, DELETE, etc.)
            rows_affected = result.get("row_count", 0)
            if rows_affected >= 0:
                print(f"Query executed successfully. Rows affected: {rows_affected}")
            else:
                print("Query executed successfully.")
    else:
        print(f"❌ Query execution failed: {result.get('error', 'Unknown error')}")
    
    print("=" * 60)

def main():
    """Main function to run the SQL generator with API execution"""
    
    # Initialize the SQL generator
    try:
        print("Initializing SQL Generator...")
        sql_generator = SQLGenerator()
        print("✅ SQL Generator ready!")
    except Exception as e:
        print(f"❌ Failed to initialize SQL generator: {e}")
        return
    
    print(f"\nAPI Endpoint: {API_ENDPOINT}")
    print("=" * 60)
    print("Natural Language to SQL Query Generator with API Execution")
    print("=" * 60)
    print("Enter your questions (type 'quit' to exit):")
    
    while True:
        try:
            # Get user input
            question = input("\n🤔 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            # Process the question
            process_question(sql_generator, question)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            print(f"❌ Error: {str(e)}")

def test_sample_questions():
    """Test with sample questions"""
    
    print("Testing with sample questions...")
    
    try:
        sql_generator = SQLGenerator()
    except Exception as e:
        print(f"Failed to initialize SQL generator: {e}")
        return
    
    sample_questions = [
        "How many policies are there?",
        "What are the total claims by status?",
        "Show me the top 5 policies by premium amount",
        "Count the number of open claims"
    ]
    
    for question in sample_questions:
        process_question(sql_generator, question)
        print("\n" + "="*60)

if __name__ == "__main__":
    # Run interactive mode
    main()
    
    # Uncomment below to run tests instead
    # test_sample_questions()
