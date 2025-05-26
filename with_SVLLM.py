import torch
import time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import gc
from contextlib import contextmanager
import asyncio
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. Configuration ---

MODEL_NAME = "defog/sqlcoder-7b-2"

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
use the logic and give accurate results 
necessary table joins based on the following rules:
- For a policy-related question, join 'fct_policy' and 'dim_policy' on 'policy_number' and 'org_id'.
- For a claims-related question, join 'fact_claims_dtl' and 'dim_claims' on 'claim_reference_id' and 'org_id'.
- Always use proper T-SQL syntax.
- Return only the SQL query without any explanatory text.
"""

# --- 2. vLLM Configuration and Setup ---

class vLLMConfig:
    """Configuration class for vLLM settings"""
    def __init__(self):
        self.model_name = MODEL_NAME
        self.tensor_parallel_size = self._get_optimal_tp_size()
        self.gpu_memory_utilization = 0.85  # Use 85% of GPU memory
        self.max_model_len = 4096  # Context length
        self.enable_prefix_caching = True  # Cache common prefixes
        self.max_num_seqs = 256  # Max sequences in batch
        self.swap_space = 4  # GB of CPU swap space
        self.enforce_eager = False  # Use CUDA graphs when possible
        
    def _get_optimal_tp_size(self):
        """Determine optimal tensor parallel size based on available GPUs"""
        if not torch.cuda.is_available():
            return 1
        
        gpu_count = torch.cuda.device_count()
        # For 7B models, usually don't need more than 2-4 GPUs
        if gpu_count >= 4:
            return 4
        elif gpu_count >= 2:
            return 2
        else:
            return 1

def setup_vllm_engine():
    """Initialize vLLM engine with optimized configuration"""
    config = vLLMConfig()
    
    print("🚀 Initializing vLLM engine...")
    print(f"📊 GPU Count: {torch.cuda.device_count()}")
    print(f"🔧 Tensor Parallel Size: {config.tensor_parallel_size}")
    print(f"💾 GPU Memory Utilization: {config.gpu_memory_utilization}")
    
    start_time = time.time()
    
    try:
        # Initialize vLLM engine
        llm = LLM(
            model=config.model_name,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            enable_prefix_caching=config.enable_prefix_caching,
            max_num_seqs=config.max_num_seqs,
            swap_space=config.swap_space,
            enforce_eager=config.enforce_eager,
            trust_remote_code=True,
            quantization=None,  # Can use "awq", "gptq", etc. if model supports
        )
        
        # Load tokenizer for preprocessing
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            use_fast=True,
            padding_side="left"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        load_time = time.time() - start_time
        print(f"⚡ vLLM engine loaded in {load_time:.2f}s")
        
        return llm, tokenizer, config
        
    except Exception as e:
        logger.error(f"Failed to initialize vLLM: {e}")
        # Fallback to smaller configuration
        print("🔄 Falling back to conservative settings...")
        llm = LLM(
            model=config.model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.7,
            max_model_len=2048,
            swap_space=2,
            trust_remote_code=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return llm, tokenizer, config

# --- 3. Optimized Prompt Construction with Caching ---

class PromptCache:
    """Enhanced prompt cache with vLLM optimizations"""
    def __init__(self):
        self.base_prompt_template = None
        self.common_prefixes = {}  # Cache for prefix caching
        self._setup_template()
    
    def _setup_template(self):
        """Pre-compile the base template"""
        self.base_prompt_template = f"""### Task
### Output Requirements

Use valid T-SQL syntax for Azure SQL Server. Avoid using PostgreSQL-specific features (like "::FLOAT"). Use CAST(... AS FLOAT) for casting.

Generate a SQL query to answer [QUESTION]{{}}[/QUESTION]

### Database Schema

The query will run on a database with the following schema:

{DATABASE_SCHEMA}

### Join Rules

{JOIN_RULES}

### Answer

Given the database schema and join rules, here is the SQL query that answers [QUESTION]{{}}[/QUESTION]

[SQL]
"""
    
    def make_prompt(self, user_question):
        """Fast prompt construction using pre-compiled template"""
        return self.base_prompt_template.format(user_question, user_question)
    
    def get_prompt_hash(self, user_question):
        """Generate hash for prompt caching"""
        return hash(user_question)

# Global prompt cache
prompt_cache = PromptCache()

# --- 4. vLLM-Optimized SQL Generation ---

class SQLGenerator:
    """High-performance SQL generator using vLLM"""
    
    def __init__(self, llm, tokenizer, config):
        self.llm = llm
        self.tokenizer = tokenizer
        self.config = config
        self.sampling_params = self._create_sampling_params()
        
    def _create_sampling_params(self):
        """Create optimized sampling parameters"""
        return SamplingParams(
            temperature=0.0,  # Deterministic for SQL generation
            top_p=1.0,
            max_tokens=200,
            stop=["[/SQL]", "\n\n", "###"],  # Stop tokens
            use_beam_search=False,  # Faster than beam search for this use case
        )
    
    def generate_sql_single(self, user_question: str) -> str:
        """Generate SQL for a single question"""
        print(f"\n🤔 Processing question: '{user_question}'")
        
        start_time = time.time()
        
        # Create prompt
        prompt = prompt_cache.make_prompt(user_question)
        
        # Generate with vLLM
        generate_start = time.time()
        outputs = self.llm.generate([prompt], self.sampling_params)
        generate_time = time.time() - generate_start
        
        # Extract SQL
        generated_text = outputs[0].outputs[0].text
        sql_query = self._extract_sql(generated_text)
        
        total_time = time.time() - start_time
        
        # Calculate performance metrics
        output_tokens = len(outputs[0].outputs[0].token_ids)
        tokens_per_second = output_tokens / generate_time if generate_time > 0 else 0
        
        print(f"⏱️  Performance: {total_time:.2f}s total ({tokens_per_second:.1f} tok/s)")
        
        return sql_query
    
    def generate_sql_batch(self, questions: List[str]) -> List[str]:
        """Generate SQL for multiple questions in batch"""
        print(f"\n📦 Processing {len(questions)} questions in batch...")
        
        start_time = time.time()
        
        # Create prompts
        prompts = [prompt_cache.make_prompt(q) for q in questions]
        
        # Generate batch with vLLM (automatic batching and optimization)
        outputs = self.llm.generate(prompts, self.sampling_params)
        
        # Extract SQL queries
        sql_queries = []
        for output in outputs:
            generated_text = output.outputs[0].text
            sql_query = self._extract_sql(generated_text)
            sql_queries.append(sql_query)
        
        total_time = time.time() - start_time
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        avg_tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        
        print(f"📊 Batch completed in {total_time:.2f}s ({avg_tokens_per_second:.1f} tok/s avg)")
        
        return sql_queries
    
    def _extract_sql(self, generated_text: str) -> str:
        """Extract SQL query from generated text"""
        sql_query = generated_text.strip()
        
        # Remove common prefixes/suffixes
        if "[SQL]" in sql_query:
            sql_query = sql_query.split("[SQL]")[-1].strip()
        
        # Clean up common artifacts
        sql_query = sql_query.replace("[/SQL]", "").strip()
        
        # Remove trailing explanations
        lines = sql_query.split('\n')
        sql_lines = []
        for line in lines:
            if line.strip() and not line.strip().startswith('--'):
                sql_lines.append(line)
            elif line.strip().startswith('--'):
                break
        
        return '\n'.join(sql_lines).strip()

# --- 5. Async Support for High Throughput ---

class AsyncSQLGenerator:
    """Async wrapper for high-throughput applications"""
    
    def __init__(self, sql_generator):
        self.sql_generator = sql_generator
    
    async def generate_sql_async(self, user_question: str) -> str:
        """Async SQL generation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.sql_generator.generate_sql_single, 
            user_question
        )
    
    async def generate_sql_batch_async(self, questions: List[str]) -> List[str]:
        """Async batch SQL generation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.sql_generator.generate_sql_batch, 
            questions
        )

# --- 6. Performance Monitoring ---

class PerformanceMonitor:
    """Monitor and log performance metrics"""
    
    def __init__(self):
        self.query_times = []
        self.batch_times = []
        self.token_rates = []
    
    def log_query(self, duration: float, tokens_per_second: float):
        """Log single query performance"""
        self.query_times.append(duration)
        self.token_rates.append(tokens_per_second)
    
    def log_batch(self, duration: float, batch_size: int):
        """Log batch performance"""
        self.batch_times.append(duration)
    
    def get_stats(self):
        """Get performance statistics"""
        if not self.query_times:
            return "No queries processed yet"
        
        avg_time = sum(self.query_times) / len(self.query_times)
        avg_tokens = sum(self.token_rates) / len(self.token_rates)
        
        return f"""
📊 Performance Statistics:
   • Average query time: {avg_time:.2f}s
   • Average token rate: {avg_tokens:.1f} tok/s
   • Total queries: {len(self.query_times)}
   • Total batches: {len(self.batch_times)}
        """

# --- 7. Memory Management ---

def cleanup_memory():
    """Enhanced memory cleanup for vLLM"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

# --- 8. Main Application ---

def main():
    """Main application with vLLM integration"""
    try:
        # Initialize vLLM engine
        llm, tokenizer, config = setup_vllm_engine()
        
        # Create SQL generator
        sql_generator = SQLGenerator(llm, tokenizer, config)
        async_generator = AsyncSQLGenerator(sql_generator)
        monitor = PerformanceMonitor()
        
        print("\n" + "="*60)
        print("🚀 vLLM-Powered SQL Generator Ready!")
        print("⚡ Enhanced performance with continuous batching")
        print("💾 Optimized memory usage with prefix caching")
        print("📦 Efficient batch processing")
        print("="*60)
        
        # Interactive loop
        while True:
            try:
                user_input = input("\n❓ Enter your question, 'batch' for batch demo, 'async' for async demo, 'stats' for performance, or 'quit' to exit: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'stats':
                    print(monitor.get_stats())
                    continue
                
                if user_input.lower() == 'batch':
                    # Batch processing demo
                    demo_questions = [
                        "Show me total claims by status",
                        "What is the average premium by country?",
                        "List all policies expiring next month",
                        "Show claims with highest paid amounts",
                        "Count total number of active policies",
                        "Find policies with premium above 10000"
                    ]
                    
                    batch_results = sql_generator.generate_sql_batch(demo_questions)
                    
                    for q, sql in zip(demo_questions, batch_results):
                        print(f"\n❓ {q}")
                        print(f"🎯 {sql}")
                    continue
                
                if user_input.lower() == 'async':
                    # Async demo
                    async def async_demo():
                        demo_questions = [
                            "Show total premium by product",
                            "List recent claims",
                            "Find expired policies"
                        ]
                        
                        # Process multiple queries concurrently
                        tasks = [async_generator.generate_sql_async(q) for q in demo_questions]
                        results = await asyncio.gather(*tasks)
                        
                        for q, sql in zip(demo_questions, results):
                            print(f"\n❓ {q}")
                            print(f"🎯 {sql}")
                    
                    asyncio.run(async_demo())
                    continue
                
                if not user_input:
                    print("⚠️  Please enter a question.")
                    continue
                
                # Generate SQL
                start_time = time.time()
                sql_query = sql_generator.generate_sql_single(user_input)
                duration = time.time() - start_time
                
                print(f"\n🎯 Generated SQL Query:")
                print("-" * 40)
                print(sql_query)
                print("-" * 40)
                
                # Log performance
                monitor.log_query(duration, 0)  # Token rate calculated in generator
                
                # Periodic cleanup
                cleanup_memory()
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error during inference: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Error initializing vLLM: {str(e)}")
        return 1
    
    finally:
        # Final cleanup
        cleanup_memory()
        print("🧹 Memory cleaned up")
    
    return 0

if __name__ == "__main__":
    # Check vLLM installation
    try:
        import vllm
        print(f"✅ vLLM version: {vllm.__version__}")
    except ImportError:
        print("❌ vLLM not installed. Please install with: pip install vllm")
        exit(1)
    
    exit_code = main()
