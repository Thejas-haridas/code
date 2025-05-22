from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging

# Check GPU availability and set device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    torch.cuda.empty_cache()  # Clear GPU cache
else:
    device = torch.device("cpu")
    print("No GPU detected, using CPU")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
Generate a tSQL query to answer this question: `{question}`.no other information is needed only the query for the azur sql server

### Database Schema
{DATABASE_SCHEMA}

### Instructions
{JOIN_RULES}

### SQL Query
```sql"""
        
        return prompt
    
    def extract_sql_from_response(self, response: str) -> str:
        """Extract SQL query from model response with improved debugging"""
        
        logger.info(f"Raw response length: {len(response)}")
        logger.info(f"Raw response preview: {response[:200]}...")
        
        # Clean up the response first
        response = response.strip()
        
        if not response:
            logger.warning("Empty response received")
            return "No SQL query generated"
        
        # Look for SQL code blocks first
        if "```sql" in response:
            sql_start = response.find("```sql") + 6
            sql_end = response.find("```", sql_start)
            if sql_end != -1:
                sql_query = response[sql_start:sql_end].strip()
                logger.info(f"Found SQL in code block: {sql_query[:100]}...")
                return sql_query
        
        # Look for SELECT statements
        if "SELECT" in response.upper():
            lines = response.split('\n')
            sql_lines = []
            capturing = False
            
            for line in lines:
                line = line.strip()
                
                # Start capturing when we see SELECT
                if "SELECT" in line.upper():
                    capturing = True
                
                if capturing:
                    # Stop if we hit common stop patterns
                    if any(stop_word in line.lower() for stop_word in [
                        'assistant', 'i am a', 'here is', 'summary', 'experience',
                        'looking for', 'software developer', 'job search', 'note:', 'explanation:'
                    ]):
                        break
                    
                    sql_lines.append(line)
                    
                    # Stop at semicolon
                    if line.endswith(';'):
                        break
            
            if sql_lines:
                sql_result = '\n'.join(sql_lines).strip()
                logger.info(f"Extracted SQL from SELECT: {sql_result[:100]}...")
                return sql_result
        
        # If no SQL found, return debug info
        logger.warning("No SQL query found in response")
        return f"Debug - Response preview: {response[:500]}..."
    
    def generate_sql(self, question: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        """Generate T-SQL query from natural language question with improved debugging"""
        
        try:
            logger.info(f"Generating SQL for question: {question}")
            
            # Create prompt
            prompt = self.create_prompt(question)
            logger.info(f"Prompt length: {len(prompt)}")
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048,
                padding=False
            )
            
            logger.info(f"Input token length: {inputs['input_ids'].shape[1]}")
            
            # Move inputs to the same device as model
            model_device = next(self.model.parameters()).device
            inputs = {k: v.to(model_device, non_blocking=True) for k, v in inputs.items()}
            
            # Generation settings
            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": True if temperature > 0 else False,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": 1.1,
                "early_stopping": False,  # Changed to False
                "use_cache": True,
                "num_beams": 1,
            }
            
            logger.info("Starting generation...")
            
            # Generate response
            with torch.no_grad():
                if torch.cuda.is_available():
                    # Use autocast for mixed precision on GPU
                    with torch.amp.autocast('cuda'):  # Updated autocast call
                        outputs = self.model.generate(
                            **inputs,
                            **generation_kwargs
                        )
                else:
                    outputs = self.model.generate(
                        **inputs,
                        **generation_kwargs
                    )
            
            logger.info("Generation completed")
            
            # Move output back to CPU for decoding if needed
            if torch.cuda.is_available():
                outputs = outputs.cpu()
            
            # Decode response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Generated text length: {len(generated_text)}")
            
            # Extract the new content (remove the prompt part)
            new_content = generated_text[len(prompt):].strip()
            logger.info(f"New content length: {len(new_content)}")
            
            # Extract SQL from the generated text
            sql_query = self.extract_sql_from_response(new_content)
            
            # Clear GPU cache after generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return sql_query
            
        except Exception as e:
            logger.error(f"Error generating SQL: {str(e)}")
            # Clear GPU cache on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return f"Error: {str(e)}"

    def get_table_info_query(self):
        """Return a query to show available tables"""
        return """SELECT 
    'dwh.dim_claims' as table_name,
    'Claims dimension table' as description
UNION ALL
SELECT 
    'dwh.dim_policy' as table_name,
    'Policy dimension table' as description
UNION ALL
SELECT 
    'dwh.fact_claims_dtl' as table_name,
    'Claims fact detail table' as description
UNION ALL
SELECT 
    'dwh.fact_premium' as table_name,
    'Premium fact table' as description
UNION ALL
SELECT 
    'dwh.fct_policy' as table_name,
    'Policy fact table' as description;"""

def main():
    """Main function to run the SQL generator with improved debugging"""
    
    # Initialize the SQL generator
    try:
        sql_generator = SQLGenerator()
    except Exception as e:
        print(f"Failed to initialize SQL generator: {e}")
        return
    
    print("T-SQL Query Generator is ready!")
    print("Enter your questions (type 'quit' to exit, 'tables' to see available tables):")
    print("-" * 50)
    
    while True:
        try:
            # Get user input
            question = input("\nQuestion: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            # Handle special commands
            if question.lower() in ['tables', 'show tables', 'what tables']:
                print("\nAvailable Tables:")
                print("-" * 30)
                print(sql_generator.get_table_info_query())
                print("-" * 30)
                continue
            
            print("Generating SQL query...")
            
            # Generate SQL
            sql_query = sql_generator.generate_sql(question)
            
            print("\nGenerated SQL:")
            print("-" * 30)
            if sql_query and sql_query.strip():
                print(sql_query)
            else:
                print("No SQL query was generated. Please try rephrasing your question.")
                print("For example: 'Count all policies' or 'Show all open claims'")
            print("-" * 30)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            print(f"Error: {str(e)}")

def test_simple_queries():
    """Test with some simple queries for debugging"""
    
    sql_generator = SQLGenerator()
    
    test_questions = [
        "SELECT COUNT(*) FROM dwh.dim_policy",
        "How many policies are there?",
        "Show me all tables",
        "Count the total policies"
    ]
    
    print("Testing simple queries...")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        try:
            sql_query = sql_generator.generate_sql(question)
            print(f"SQL:\n{sql_query}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 30)

if __name__ == "__main__":
    # Uncomment the line below to run interactive mode
    main()
    
    # Uncomment the line below to test simple queries
    # test_simple_queries()
