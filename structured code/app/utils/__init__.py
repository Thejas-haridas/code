"""
Utility functions for the RAG-Enhanced SQL Query Generator and Analyzer.

This module contains helper functions for device management, model loading,
text processing, SQL cleaning, memory management, and file operations.
"""

import torch
import time
import gc
import os
import json
import logging
import numpy as np
import requests
from contextlib import contextmanager
from typing import Tuple, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

logger = logging.getLogger(__name__)

# --- Device and Hardware Management ---
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

def cleanup_memory():
    """Aggressive memory cleanup."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

# --- Model Loading Functions ---
def load_sql_model_and_tokenizer(model_name: str, device: torch.device) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load SQL generation model and tokenizer with optimizations."""
    logger.info("🔄 Loading SQL model and tokenizer...")
    load_start_time = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side="left")
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
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    
    if hasattr(model, 'config'):
        model.config.use_cache = True
    
    logger.info(f"🤖 SQL model loaded in {time.time() - load_start_time:.2f}s")
    return model, tokenizer

def load_analysis_model_and_tokenizer(model_name: str, device: torch.device) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load Phi-3-mini analysis model and tokenizer with optimizations."""
    logger.info("🔄 Loading Phi-3-mini analysis model and tokenizer...")
    load_start_time = time.time()
    
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    
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
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    
    if hasattr(model, 'config'):
        model.config.use_cache = True
    
    logger.info(f"🤖 Phi-3-mini analysis model loaded in {time.time() - load_start_time:.2f}s")
    return model, tokenizer

# --- Text Generation and Processing ---
@contextmanager
def inference_mode():
    """Context manager for optimized inference."""
    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                yield
        else:
            yield

def generate_text_optimized(prompt: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                          max_new_tokens: int, temperature: float = 0.0) -> str:
    """Highly optimized text generation function."""
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
            'use_cache': True,
            'early_stopping': True,
        }
        
        if temperature > 0:
            gen_kwargs['temperature'] = temperature
            
        outputs = model.generate(**gen_kwargs)
    
    new_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Cleanup
    del inputs, outputs, gen_kwargs
    cleanup_memory()
    
    return generated_text

# --- SQL Processing Functions ---
def extract_sql_from_response(response: str) -> str:
    """Extracts SQL code from the model's response with improved T-SQL parsing."""
    if "```sql" in response:
        start = response.find("```sql") + 6
        end = response.find("```", start)
        if end != -1:
            sql_query = response[start:end].strip()
            return clean_tsql_query(sql_query)
    
    markers = ["### T-SQL Query", "T-SQL Query:", "Query:", "SELECT", "WITH", "INSERT", "UPDATE", "DELETE"]
    for marker in markers:
        if marker in response:
            start = response.find(marker)
            if marker in ["SELECT", "WITH", "INSERT", "UPDATE", "DELETE"]:
                sql_part = response[start:]
            else:
                sql_part = response[start + len(marker):]
            
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
                    break
            
            if sql_lines:
                return clean_tsql_query('\n'.join(sql_lines))
    
    return clean_tsql_query(response.strip())

def clean_tsql_query(sql_query: str) -> str:
    """Clean and validate T-SQL query for common issues."""
    if not sql_query:
        return sql_query
    
    lines = sql_query.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.rstrip()
        if not line.strip():
            cleaned_lines.append(line)
            continue
        if line.strip().startswith('#'):
            continue
        
        line_upper = line.upper()
        if 'LIMIT ' in line_upper and 'SELECT' in line_upper:
            line = line.replace('LIMIT ', '-- LIMIT converted to TOP: ')
        
        if 'NULLS FIRST' in line_upper or 'NULLS LAST' in line_upper:
            line = line.replace('NULLS FIRST', '').replace('NULLS LAST', '')
            line = line.replace('nulls first', '').replace('nulls last', '')
        
        cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines).strip()
    
    # Remove double semicolons
    while result.endswith(';;'):
        result = result[:-1]
    
    return result

# --- API and Execution Functions ---
def execute_sql(sql_query: str, api_endpoint: str) -> Tuple[Dict[str, Any], float]:
    """Executes SQL query via the external API."""
    start_time = time.time()
    
    try:
        response = requests.post(
            api_endpoint,
            json={"query": sql_query},
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as e:
        logger.error(f"SQL execution API error: {e}")
        result = {
            "error": str(e), 
            "status_code": getattr(e.response, 'status_code', 500) if hasattr(e, 'response') and e.response else 500
        }
    
    execution_time = time.time() - start_time
    logger.info(f"SQL executed in {execution_time:.2f}s")
    return result, execution_time

# --- File Operations ---
def save_result_to_file(data: dict, save_path: str) -> str:
    """Saves the full transaction to a timestamped file."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(save_path, f"query_{timestamp}.json")
    
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4, default=str)
        logger.info(f"Result saved to {filename}")
        return filename
    except IOError as e:
        logger.error(f"Failed to save file: {e}")
        return f"Error saving file: {e}"

def ensure_directories_exist(*paths):
    """Ensure that all provided directory paths exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)

# --- Prompt Engineering Functions ---
def construct_rag_prompt(question: str, relevant_tables: list, join_conditions: str, tsql_rules: str) -> str:
    """Creates a structured prompt using retrieved schema elements."""
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

{join_conditions}

{tsql_rules}

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

def make_analysis_prompt(question: str, sql_query: str, sql_result: dict) -> str:
    """Creates a streamlined prompt for Phi-3-mini to analyze SQL results."""
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

# --- Memory and System Monitoring ---
def get_memory_info() -> dict:
    """Get current memory usage statistics."""
    memory_info = {}
    
    if torch.cuda.is_available():
        memory_info["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1e9
        memory_info["gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1e9
        memory_info["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    try:
        import psutil
        process = psutil.Process()
        memory_info["cpu_memory_mb"] = process.memory_info().rss / 1024 / 1024
    except ImportError:
        logger.warning("psutil not available for CPU memory monitoring")
    
    return memory_info

# Export all utility functions
__all__ = [
    "setup_device",
    "cleanup_memory",
    "load_sql_model_and_tokenizer",
    "load_analysis_model_and_tokenizer",
    "inference_mode",
    "generate_text_optimized",
    "extract_sql_from_response",
    "clean_tsql_query",
    "execute_sql",
    "save_result_to_file",
    "ensure_directories_exist",
    "construct_rag_prompt",
    "make_analysis_prompt",
    "get_memory_info"
]