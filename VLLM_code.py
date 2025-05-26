import time
import gc
import os
from typing import List, Optional
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
import asyncio
import torch

# Set environment variables to handle compilation issues
os.environ['TORCH_LOGS'] = "+dynamo"
os.environ['TORCHDYNAMO_VERBOSE'] = "1"

# Fallback to disable Triton if compilation fails
def set_fallback_env():
    """Set environment variables for fallback options"""
    os.environ['VLLM_USE_TRITON'] = '0'
    torch._dynamo.config.suppress_errors = True
    print("🔧 Fallback configuration applied")

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

# --- 2. Optimized Prompt Construction ---

class PromptCache:
    """Cache for reusable prompt components"""
    def __init__(self):
        self.base_prompt_template = None
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
    
    def make_prompt(self, user_question: str) -> str:
        """Fast prompt construction using pre-compiled template"""
        return self.base_prompt_template.format(user_question, user_question)

# Global prompt cache
prompt_cache = PromptCache()

# --- 3. vLLM Engine Setup ---

class OptimizedSQLGenerator:
    """High-performance SQL generator using vLLM"""
    
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.llm = None
        self.async_engine = None
        self.sampling_params = None
        self._setup_sampling_params()
    
    def _setup_sampling_params(self):
        """Configure optimized sampling parameters"""
        self.sampling_params = SamplingParams(
            temperature=0.0,  # Deterministic output
            top_p=1.0,
            max_tokens=200,
            stop=["[/SQL]", "\n\n", "```"],  # Stop tokens
            skip_special_tokens=True,
        )
    
    def load_model(self, **kwargs):
        """Load vLLM model with progressive fallback configurations"""
        print("\n🔄 Loading vLLM model...")
        load_start = time.time()
        
        # Configuration attempts in order of preference
        configs = [
            # Config 1: High performance
            {
                "model": self.model_name,
                "tensor_parallel_size": 1,
                "dtype": "float16",
                "max_model_len": 2048,
                "gpu_memory_utilization": 0.8,
                "swap_space": 4,
                "trust_remote_code": True,
                "download_dir": "./model_cache",
                "name": "High Performance"
            },
            # Config 2: Conservative memory
            {
                "model": self.model_name,
                "tensor_parallel_size": 1,
                "dtype": "float16",
                "max_model_len": 1024,
                "gpu_memory_utilization": 0.6,
                "swap_space": 2,
                "trust_remote_code": True,
                "download_dir": "./model_cache",
                "enforce_eager": True,
                "name": "Conservative Memory"
            },
            # Config 3: Minimal requirements
            {
                "model": self.model_name,
                "tensor_parallel_size": 1,
                "dtype": "auto",
                "max_model_len": 512,
                "gpu_memory_utilization": 0.4,
                "swap_space": 1,
                "trust_remote_code": True,
                "download_dir": "./model_cache",
                "enforce_eager": True,
                "disable_custom_all_reduce": True,
                "name": "Minimal Requirements"
            },
            # Config 4: CPU fallback
            {
                "model": self.model_name,
                "tensor_parallel_size": 1,
                "dtype": "float32",
                "max_model_len": 512,
                "gpu_memory_utilization": 0.3,
                "swap_space": 8,
                "trust_remote_code": True,
                "download_dir": "./model_cache",
                "enforce_eager": True,
                "disable_custom_all_reduce": True,
                "device": "cpu",
                "name": "CPU Fallback"
            }
        ]
        
        # Try each configuration
        for i, config in enumerate(configs):
            config_name = config.pop("name", f"Config {i+1}")
            
            # Override with user-provided kwargs
            final_config = {**config, **kwargs}
            
            try:
                print(f"🔧 Trying {config_name} configuration...")
                
                # Apply environment variables for problematic configs
                if i >= 1:  # Apply fallback settings for configs 2+
                    set_fallback_env()
                
                self.llm = LLM(**final_config)
                
                load_time = time.time() - load_start
                print(f"✅ vLLM model loaded with {config_name} in {load_time:.2f}s")
                print(f"🤖 Model: {self.model_name}")
                print(f"💾 GPU memory utilization: {final_config.get('gpu_memory_utilization', 0.8)*100}%")
                print(f"📏 Max sequence length: {final_config.get('max_model_len', 2048)}")
                
                if i > 0:
                    print(f"⚠️  Running in {config_name.lower()} mode")
                
                return  # Success, exit the function
                
            except Exception as e:
                print(f"❌ {config_name} failed: {str(e)[:100]}...")
                
                # Clean up any partial initialization
                if hasattr(self, 'llm') and self.llm is not None:
                    try:
                        del self.llm
                        self.llm = None
                    except:
                        pass
                
                # Clean up GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Continue to next configuration
                continue
        
        # If all configurations failed
        raise RuntimeError(f"Failed to load model with any configuration. Last error: {str(e)}")
    
    async def load_async_model(self, **kwargs):
        """Load async vLLM engine for concurrent processing"""
        print("\n🔄 Loading async vLLM engine...")
        load_start = time.time()
        
        # Async engine configuration
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            tensor_parallel_size=1,
            dtype="float16",
            max_model_len=2048,
            gpu_memory_utilization=0.8,
            trust_remote_code=True,
            download_dir="./model_cache",
            **kwargs
        )
        
        try:
            self.async_engine = AsyncLLMEngine.from_engine_args(engine_args)
            load_time = time.time() - load_start
            print(f"✅ Async vLLM engine loaded in {load_time:.2f}s")
        except Exception as e:
            print(f"⚠️  Async load failed, trying fallback...")
            set_fallback_env()
            
            # Fallback for async
            fallback_args = AsyncEngineArgs(
                model=self.model_name,
                tensor_parallel_size=1,
                dtype="auto",
                max_model_len=1024,
                gpu_memory_utilization=0.6,
                trust_remote_code=True,
                download_dir="./model_cache",
                enforce_eager=True,
                **kwargs
            )
            
            try:
                self.async_engine = AsyncLLMEngine.from_engine_args(fallback_args)
                load_time = time.time() - load_start
                print(f"✅ Async vLLM engine loaded with fallback in {load_time:.2f}s")
            except Exception as e2:
                print(f"❌ Failed to load async vLLM engine: {e2}")
                raise
    
    def generate_sql(self, user_question: str) -> str:
        """Generate SQL using synchronous vLLM"""
        if self.llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        print(f"\n🤔 Processing question: '{user_question}'")
        start_time = time.time()
        
        try:
            # Prepare prompt
            prompt = prompt_cache.make_prompt(user_question)
            
            # Generate with vLLM
            outputs = self.llm.generate([prompt], self.sampling_params)
            
            # Extract SQL
            generated_text = outputs[0].outputs[0].text.strip()
            sql_query = self._extract_sql(generated_text)
            
            # Performance metrics
            total_time = time.time() - start_time
            tokens_generated = len(outputs[0].outputs[0].token_ids)
            tokens_per_second = tokens_generated / total_time if total_time > 0 else 0
            
            print(f"⏱️  Generated in {total_time:.2f}s ({tokens_per_second:.1f} tok/s)")
            
            return sql_query
            
        except Exception as e:
            print(f"❌ Error during generation: {e}")
            return f"-- Error generating SQL: {str(e)}"
    
    def generate_sql_batch(self, questions: List[str]) -> List[str]:
        """Generate SQL for multiple questions in a single batch"""
        if self.llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        print(f"\n📦 Processing batch of {len(questions)} questions...")
        start_time = time.time()
        
        try:
            # Prepare prompts
            prompts = [prompt_cache.make_prompt(q) for q in questions]
            
            # Generate batch with vLLM (automatically optimized)
            outputs = self.llm.generate(prompts, self.sampling_params)
            
            # Extract SQL queries
            sql_queries = []
            for output in outputs:
                generated_text = output.outputs[0].text.strip()
                sql_query = self._extract_sql(generated_text)
                sql_queries.append(sql_query)
            
            # Performance metrics
            total_time = time.time() - start_time
            total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
            tokens_per_second = total_tokens / total_time if total_time > 0 else 0
            throughput = len(questions) / total_time if total_time > 0 else 0
            
            print(f"⚡ Batch completed in {total_time:.2f}s")
            print(f"📈 Throughput: {throughput:.1f} queries/s, {tokens_per_second:.1f} tok/s")
            
            return sql_queries
            
        except Exception as e:
            print(f"❌ Error during batch generation: {e}")
            return [f"-- Error: {str(e)}" for _ in questions]
    
    async def generate_sql_async(self, questions: List[str]) -> List[str]:
        """Generate SQL asynchronously for maximum concurrency"""
        if self.async_engine is None:
            raise RuntimeError("Async engine not loaded. Call load_async_model() first.")
        
        print(f"\n🚀 Processing {len(questions)} questions asynchronously...")
        start_time = time.time()
        
        try:
            # Prepare prompts
            prompts = [prompt_cache.make_prompt(q) for q in questions]
            
            # Generate all requests concurrently
            tasks = []
            for i, prompt in enumerate(prompts):
                task = self.async_engine.generate(
                    prompt, 
                    self.sampling_params, 
                    request_id=f"sql_gen_{i}"
                )
                tasks.append(task)
            
            # Wait for all completions
            results = await asyncio.gather(*tasks)
            
            # Extract SQL queries
            sql_queries = []
            for result in results:
                generated_text = result.outputs[0].text.strip()
                sql_query = self._extract_sql(generated_text)
                sql_queries.append(sql_query)
            
            # Performance metrics
            total_time = time.time() - start_time
            throughput = len(questions) / total_time if total_time > 0 else 0
            
            print(f"🚀 Async batch completed in {total_time:.2f}s")
            print(f"📈 Throughput: {throughput:.1f} queries/s")
            
            return sql_queries
            
        except Exception as e:
            print(f"❌ Error during async generation: {e}")
            return [f"-- Error: {str(e)}" for _ in questions]
    
    def _extract_sql(self, generated_text: str) -> str:
        """Extract clean SQL from generated text"""
        sql_query = generated_text.strip()
        
        # Remove common prefixes/suffixes
        if "[SQL]" in sql_query:
            sql_query = sql_query.split("[SQL]")[-1].strip()
        
        if "```sql" in sql_query.lower():
            sql_query = sql_query.split("```sql")[-1].split("```")[0].strip()
        
        if "```" in sql_query:
            sql_query = sql_query.split("```")[0].strip()
        
        return sql_query

# --- 4. Performance Utilities ---

def cleanup_memory():
    """Clean up memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("🧹 Memory cleaned up")

def check_system_requirements():
    """Check system requirements and suggest fixes"""
    print("🔍 Checking system requirements...")
    
    # Check CUDA
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1e9
            print(f"✅ GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"💾 Memory: {memory_gb:.1f} GB")
            
            # Check available memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated(i) / 1e9
                cached = torch.cuda.memory_reserved(i) / 1e9
                free = memory_gb - allocated - cached
                print(f"🆓 Available: ~{free:.1f} GB")
                
                if memory_gb < 4:
                    print("⚠️  Warning: GPU has less than 4GB memory")
                    print("💡 Consider using CPU mode or a smaller model")
                elif memory_gb < 8:
                    print("⚠️  Warning: GPU memory is limited")
                    print("💡 Will use conservative memory settings")
    else:
        print("⚠️  CUDA not available - will try CPU mode")
    
    # Check compiler
    try:
        import subprocess
        result = subprocess.run(['gcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ GCC compiler available")
        else:
            print("⚠️  GCC compiler not found")
            print("💡 Install with: apt update && apt install -y build-essential")
    except:
        print("❌ No C compiler found")
        print("💡 Install with: apt update && apt install -y build-essential")
    
    # Check available RAM
    try:
        import psutil
        memory = psutil.virtual_memory()
        total_gb = memory.total / 1e9
        available_gb = memory.available / 1e9
        print(f"🧠 System RAM: {total_gb:.1f} GB (available: {available_gb:.1f} GB)")
        
        if available_gb < 4:
            print("⚠️  Warning: Low system memory available")
            print("💡 Close other applications to free memory")
    except ImportError:
        print("💡 Install psutil for memory checking: pip install psutil")
    
    print()

# --- 5. Main Application ---

def main():
    """Main application with vLLM integration"""
    try:
        # Check system requirements
        check_system_requirements()
        
        # Initialize SQL generator
        sql_generator = OptimizedSQLGenerator()
        
        # Load model with optimizations
        sql_generator.load_model()
        
        print("\n" + "="*60)
        print("🚀 vLLM SQL Generator Ready!")
        print("💡 High-performance inference enabled")
        print("="*60)
        
        # Interactive loop
        while True:
            try:
                user_input = input("\n❓ Enter your question, 'batch' for batch demo, 'async' for async demo, or 'quit': ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'batch':
                    # Batch processing demo
                    demo_questions = [
                        "Show me total claims by status",
                        "What is the average premium by country?",
                        "List all policies expiring next month",
                        "Show claims with highest paid amounts",
                        "What is the total sum insured by product?",
                        "Show premium trends by month",
                        "List open claims older than 6 months",
                        "What is the claim frequency by line of business?"
                    ]
                    
                    results = sql_generator.generate_sql_batch(demo_questions)
                    
                    print("\n📊 Batch Results:")
                    for i, (q, sql) in enumerate(zip(demo_questions, results), 1):
                        print(f"\n{i}. ❓ {q}")
                        print(f"   🎯 {sql}")
                    continue
                
                if user_input.lower() == 'async':
                    # Async processing demo
                    async def async_demo():
                        await sql_generator.load_async_model()
                        demo_questions = [
                            "Show me total claims by status",
                            "What is the average premium by country?",
                            "List all policies expiring next month",
                            "Show claims with highest paid amounts"
                        ]
                        results = await sql_generator.generate_sql_async(demo_questions)
                        
                        print("\n🚀 Async Results:")
                        for i, (q, sql) in enumerate(zip(demo_questions, results), 1):
                            print(f"\n{i}. ❓ {q}")
                            print(f"   🎯 {sql}")
                    
                    # Run async demo
                    asyncio.run(async_demo())
                    continue
                
                if not user_input:
                    print("⚠️  Please enter a question.")
                    continue
                
                # Single query generation
                sql_query = sql_generator.generate_sql(user_input)
                
                print(f"\n🎯 Generated SQL Query:")
                print("-" * 50)
                print(sql_query)
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error during inference: {str(e)}")
                continue
                
    except Exception as e:
        print(f"❌ Error initializing vLLM: {str(e)}")
        return 1
    
    finally:
        cleanup_memory()
    
    return 0

if __name__ == "__main__":
    exit_code = main()
