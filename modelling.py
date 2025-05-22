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
Generate a SQL query to answer this question: `{question}`

### Database Schema
{DATABASE_SCHEMA}

### Instructions
{JOIN_RULES}

### SQL Query
```sql"""
        
        return prompt
    
    def extract_sql_from_response(self, response: str) -> str:
        """Extract SQL query from model response"""
        
        # Clean up the response first
        response = response.strip()
        
        # Look for SQL code blocks
        if "```sql" in response:
            sql_start = response.find("```sql") + 6
            sql_end = response.find("```", sql_start)
            if sql_end != -1:
                return response[sql_start:sql_end].strip()
        
        # Look for SELECT statements and stop at common endpoints
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
                        'looking for', 'software developer', 'job search'
                    ]):
                        break
                    
                    sql_lines.append(line)
                    
                    # Stop at semicolon
                    if line.endswith(';'):
                        break
            
            sql_result = '\n'.join(sql_lines).strip()
            
            # Additional cleanup - remove any trailing non-SQL content
            sql_result = self.clean_trailing_content(sql_result)
            
            return sql_result
        
        # Return cleaned response
        return self.clean_trailing_content(response)
    
    def clean_trailing_content(self, text: str) -> str:
        """Remove unwanted trailing content from SQL"""
        
        # Split by common stop patterns and take only the first part
        stop_patterns = [
            'assistant', 'i am a', 'here is', 'summary', 'experience',
            'looking for', 'software developer', 'job search', 'degree in'
        ]
        
        text_lower = text.lower()
        earliest_stop = len(text)
        
        for pattern in stop_patterns:
            pos = text_lower.find(pattern)
            if pos != -1 and pos < earliest_stop:
                earliest_stop = pos
        
        if earliest_stop < len(text):
            text = text[:earliest_stop].strip()
        
        # Remove any incomplete lines at the end
        lines = text.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not any(word in line.lower() for word in stop_patterns):
                clean_lines.append(line)
            else:
                break
        
        return '\n'.join(clean_lines).strip()
    
    def generate_sql(self, question: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        """Generate T-SQL query from natural language question with GPU optimization"""
        
        try:
            # Create prompt
            prompt = self.create_prompt(question)
            
            # Tokenize input with GPU-optimized settings
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048,
                padding=False  # Don't pad for single input
            )
            
            # Move inputs to the same device as model (GPU if available)
            model_device = next(self.model.parameters()).device
            inputs = {k: v.to(model_device, non_blocking=True) for k, v in inputs.items()}
            
            # Optimize generation settings for GPU
            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": True if temperature > 0 else False,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": 1.1,
                "early_stopping": True,
                "use_cache": True,
            }
            
            # Add GPU-specific optimizations
            if torch.cuda.is_available():
                generation_kwargs.update({
                    "num_beams": 1,  # Faster on GPU
                    "length_penalty": 1.0,
                })
            
            # Generate response with GPU optimization
            with torch.no_grad():
                if torch.cuda.is_available():
                    # Use autocast for mixed precision on GPU
                    with torch.cuda.amp.autocast():
                        outputs = self.model.generate(
                            **inputs,
                            **generation_kwargs
                        )
                else:
                    outputs = self.model.generate(
                        **inputs,
                        **generation_kwargs
                    )
            
            # Move output back to CPU for decoding if needed
            if torch.cuda.is_available():
                outputs = outputs.cpu()
            
            # Decode response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract SQL from the generated text
            sql_query = self.extract_sql_from_response(generated_text[len(prompt):])
            
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

def main():
    """Main function to run the SQL generator"""
    
    # Initialize the SQL generator
    sql_generator = SQLGenerator()
    
    print("T-SQL Query Generator is ready!")
    print("Enter your questions (type 'quit' to exit):")
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
            
            print("Generating SQL query...")
            
            # Generate SQL
            sql_query = sql_generator.generate_sql(question)
            
            print("\nGenerated SQL:")
            print("-" * 30)
            print(sql_query)
            print("-" * 30)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

# Example usage function
def generate_example_queries():
    """Generate some example queries"""
    
    sql_generator = SQLGenerator()
    
    example_questions = [
        "Show me all policies with their total premium amount",
        "Find all claims that are still open",
        "What is the average claim amount by cause of loss?",
        "Show me policies that expire this year",
        "Find the top 10 highest value claims"
    ]
    
    print("Generating example queries...")
    print("=" * 50)
    
    for i, question in enumerate(example_questions, 1):
        print(f"\n{i}. Question: {question}")
        sql_query = sql_generator.generate_sql(question)
        print(f"SQL:\n{sql_query}")
        print("-" * 30)

if __name__ == "__main__":
    # Uncomment the line below to run interactive mode
    main()
    
    # Uncomment the line below to generate example queries
    # generate_example_queries()
