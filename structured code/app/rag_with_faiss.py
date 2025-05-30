"""
RAG System with FAISS-powered Schema Retrieval
"""
import torch
import time
import json
import os
import logging
import numpy as np
from contextlib import contextmanager
from typing import Tuple, Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import faiss
import requests

from utils import cleanup_memory, extract_sql_from_response, clean_tsql_query, save_result_to_file
from schema import JOIN_CONDITIONS, TSQL_RULES

# Configuration
SQL_MODEL_NAME = "defog/llama-3-sqlcoder-8b"
ANALYSIS_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
API_ENDPOINT = "http://172.200.64.182:7860/execute"
SAVE_PATH = "/home/text_sql"
EMBEDDINGS_PATH = "/home/text_sql/embeddings"

# Ensure directories exist
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(EMBEDDINGS_PATH, exist_ok=True)

logger = logging.getLogger(__name__)

class SchemaRetriever:
    """FAISS-powered schema retrieval system."""
    
    def __init__(self, table_chunks: List[Dict], embedding_model_name: str):
        self.table_chunks = table_chunks
        self.embedding_model_name = embedding_model_name
        self.embeddings_file = os.path.join(EMBEDDINGS_PATH, "table_embeddings.npy")
        self.metadata_file = os.path.join(EMBEDDINGS_PATH, "table_metadata.json")
        self.faiss_index_file = os.path.join(EMBEDDINGS_PATH, "faiss_index.index")
        self.embeddings = None
        self.faiss_index = None
        self.embedding_model = None
        self._is_initialized = False

    def load_embedding_model(self):
        """Load the sentence transformer model for embeddings."""
        if self.embedding_model is None:
            logger.info(f"🔄 Loading embedding model: {self.embedding_model_name}")
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info("✅ Embedding model loaded successfully")

    def _initialize_if_needed(self):
        """Initialize the retriever if not already done."""
        if not self._is_initialized:
            if not self.load_embeddings():
                self.create_embeddings()
            self._is_initialized = True

    def create_embeddings(self):
        """Create embeddings for all table chunks and build a FAISS index."""
        logger.info("🔄 Creating embeddings for table chunks...")
        self.load_embedding_model()

        # Prepare texts for embedding
        texts_to_embed = []
        for chunk in self.table_chunks:
            combined_text = f"{chunk['text']} {chunk['description']} {' '.join(chunk['keywords'])}"
            texts_to_embed.append(combined_text)

        # Generate embeddings
        embeddings = self.embedding_model.encode(texts_to_embed)
        self.embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # Normalize
        
        # Save embeddings
        np.save(self.embeddings_file, self.embeddings)

        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        self.faiss_index.add(self.embeddings.astype('float32'))
        
        # Save FAISS index
        faiss.write_index(self.faiss_index, self.faiss_index_file)

        # Save metadata
        with open(self.metadata_file, 'w') as f:
            json.dump({
                "table_names": [chunk["table"] for chunk in self.table_chunks],
                "embedding_dim": self.embeddings.shape[1],
                "num_tables": len(self.table_chunks)
            }, f)

        logger.info(f"✅ Created and saved embeddings for {len(self.table_chunks)} tables")

    def load_embeddings(self):
        """Load existing embeddings and FAISS index from files."""
        try:
            if (os.path.exists(self.embeddings_file) and 
                os.path.exists(self.metadata_file) and 
                os.path.exists(self.faiss_index_file)):
                
                self.embeddings = np.load(self.embeddings_file)
                self.faiss_index = faiss.read_index(self.faiss_index_file)
                
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                logger.info(f"✅ Loaded embeddings and FAISS index for {metadata['num_tables']} tables")
                return True
        except Exception as e:
            logger.warning(f"Failed to load existing embeddings: {e}")
            # Clean up partial files
            for file_path in [self.embeddings_file, self.metadata_file, self.faiss_index_file]:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except:
                        pass
        return False

    def retrieve_relevant_tables(self, question: str, top_k: int = 3) -> List[Dict]:
        """Retrieve top-k most relevant tables using FAISS."""
        self._initialize_if_needed()
        self.load_embedding_model()

        # Create embedding for the question
        question_embedding = self.embedding_model.encode([question])
        question_embedding_normalized = question_embedding / np.linalg.norm(question_embedding, axis=1, keepdims=True)

        # Perform similarity search using FAISS
        try:
            D, I = self.faiss_index.search(question_embedding_normalized.astype('float32'), k=min(top_k, len(self.table_chunks)))
            
            # Return relevant table chunks
            relevant_tables = [self.table_chunks[i] for i in I[0] if i < len(self.table_chunks)]
            logger.info(f"Retrieved {len(relevant_tables)} relevant tables: {[t['table'] for t in relevant_tables]}")
            return relevant_tables
        except Exception as e:
            logger.error(f"Error during FAISS search: {e}")
            # Fallback to simple keyword matching
            return self._fallback_retrieval(question, top_k)

    def _fallback_retrieval(self, question: str, top_k: int = 3) -> List[Dict]:
        """Fallback retrieval method using simple keyword matching."""
        logger.info("Using fallback keyword-based retrieval")
        question_lower = question.lower()
        scores = []
        
        for i, chunk in enumerate(self.table_chunks):
            score = 0
            # Check keywords
            for keyword in chunk['keywords']:
                if keyword.lower() in question_lower:
                    score += 2
            
            # Check table name
            table_name_parts = chunk['table'].split('.')[-1].split('_')
            for part in table_name_parts:
                if part.lower() in question_lower:
                    score += 1
            
            # Check description
            desc_words = chunk['description'].lower().split()
            for word in desc_words:
                if len(word) > 3 and word in question_lower:
                    score += 0.5
            
            scores.append((score, i))
        
        # Sort by score and return top_k
        scores.sort(reverse=True, key=lambda x: x[0])
        relevant_tables = [self.table_chunks[i] for _, i in scores[:top_k]]
        logger.info(f"Fallback retrieved {len(relevant_tables)} tables: {[t['table'] for t in relevant_tables]}")
        return relevant_tables


class RAGSystem:
    """Main RAG System coordinating all components."""
    
    def __init__(self, table_chunks: List[Dict], embedding_model_name: str, device: torch.device):
        self.device = device
        self.retriever = SchemaRetriever(table_chunks, embedding_model_name)
        self.sql_model = None
        self.sql_tokenizer = None
        self.analysis_model = None
        self.analysis_tokenizer = None
        self.table_chunks = table_chunks

    async def initialize(self):
        """Initialize all models."""
        self.sql_model, self.sql_tokenizer = self._load_sql_model()
        self.analysis_model, self.analysis_tokenizer = self._load_analysis_model()

    def _load_sql_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
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

    def _load_analysis_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load Phi-3-mini analysis model and tokenizer with optimizations."""
        logger.info("🔄 Loading Phi-3-mini analysis model and tokenizer...")
        load_start_time = time.time()
        
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
        
        logger.info(f"🤖 Phi-3-mini analysis model loaded in {time.time() - load_start_time:.2f}s")
        return model, tokenizer

    @contextmanager
    def inference_mode(self):
        """Context manager for optimized inference."""
        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    yield
            else:
                yield

    def generate_text_optimized(self, prompt: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, max_new_tokens: int, temperature: float = 0.0) -> str:
        """Highly optimized text generation function."""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device, non_blocking=True) for k, v in inputs.items()}
        input_length = inputs['input_ids'].shape[1]
        
        with self.inference_mode():
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
        del inputs, outputs, gen_kwargs
        cleanup_memory()
        return generated_text

    def construct_rag_prompt(self, question: str, relevant_tables: List[Dict]) -> str:
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

    def generate_sql_with_rag(self, question: str) -> Tuple[str, List[str], float]:
        """Generates SQL query using RAG approach with schema retrieval."""
        start_time = time.time()
        
        relevant_tables = self.retriever.retrieve_relevant_tables(question, top_k=3)
        retrieved_table_names = [table['table'] for table in relevant_tables]
        
        prompt = self.construct_rag_prompt(question, relevant_tables)
        response = self.generate_text_optimized(prompt, self.sql_model, self.sql_tokenizer, max_new_tokens=300)
        sql_query = extract_sql_from_response(response)
        
        generation_time = time.time() - start_time
        logger.info(f"SQL generated with RAG in {generation_time:.2f}s using tables: {retrieved_table_names}")
        return sql_query, retrieved_table_names, generation_time

    def make_analysis_prompt(self, question: str, sql_query: str, sql_result: dict) -> str:
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

    def analyze_results(self, question: str, sql_query: str, sql_result: dict) -> Tuple[str, float]:
        """Generates a natural language analysis using Phi-3-mini."""
        start_time = time.time()
        prompt = self.make_analysis_prompt(question, sql_query, sql_result)
        analysis = self.generate_text_optimized(
            prompt, 
            self.analysis_model, 
            self.analysis_tokenizer, 
            max_new_tokens=150,
            temperature=0.1
        )
        analysis_time = time.time() - start_time
        logger.info(f"Analysis generated in {analysis_time:.2f}s using Phi-3-mini")
        return analysis.strip(), analysis_time

    def execute_sql(self, sql_query: str) -> Tuple[Dict[str, Any], float]:
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

    async def process_complete_query(self, question: str, loop, executor) -> Dict[str, Any]:
        """Process complete query including SQL generation, execution, and analysis."""
        total_start_time = time.time()
        
        # Generate SQL using RAG
        sql_query, retrieved_tables, sql_generation_time = await loop.run_in_executor(
            executor, self.generate_sql_with_rag, question
        )
        
        if not sql_query:
            raise ValueError("Failed to generate SQL query.")
        
        # Execute SQL
        sql_execution_result, sql_execution_time = await loop.run_in_executor(
            executor, self.execute_sql, sql_query
        )
        
        # Generate analysis
        llm_analysis, llm_analysis_time = await loop.run_in_executor(
            executor, self.analyze_results, question, sql_query, sql_execution_result
        )
        
        total_processing_time = time.time() - total_start_time
        
        # Determine success
        is_successful = (
            sql_execution_result.get("success", False) or
            (sql_execution_result.get("data") is not None and "error" not in sql_execution_result)
        )
        
        # Prepare response
        response_data = {
            "success": is_successful,
            "question": question,
            "retrieved_tables": retrieved_tables,
            "generated_sql": sql_query,
            "sql_execution_result": sql_execution_result,
            "llm_analysis": llm_analysis,
            "retrieval_time": round(sql_generation_time, 2),
            "sql_generation_time": round(sql_generation_time, 2),
            "llm_analysis_time": round(llm_analysis_time, 2),
            "sql_execution_time": round(sql_execution_time, 2),
            "total_processing_time": round(total_processing_time, 2),
        }
        
        # Save to file
        file_saved = await loop.run_in_executor(executor, save_result_to_file, response_data)
        response_data["file_saved"] = file_saved
        
        return response_data

    async def generate_sql_only(self, question: str, loop, executor) -> Dict[str, Any]:
        """Generate SQL query without execution or analysis using RAG."""
        sql_query, retrieved_tables, generation_time = await loop.run_in_executor(
            executor, self.generate_sql_with_rag, question
        )
        
        return {
            "question": question,
            "retrieved_tables": retrieved_tables,
            "generated_sql": sql_query,
            "generation_time": round(generation_time, 2)
        }

    async def test_retrieval(self, question: str) -> Dict[str, Any]:
        """Test the RAG retrieval system independently."""
        relevant_tables = self.retriever.retrieve_relevant_tables(question, top_k=3)
        return {
            "question": question,
            "retrieved_tables": [
                {
                    "table": table["table"],
                    "description": table["description"],
                    "keywords": table["keywords"]
                }
                for table in relevant_tables
            ]
        }

    def are_models_loaded(self) -> bool:
        """Check if all models are loaded."""
        return all([
            self.sql_model is not None,
            self.sql_tokenizer is not None,
            self.analysis_model is not None,
            self.analysis_tokenizer is not None
        ])

    def get_memory_status(self) -> Dict[str, Any]:
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

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "sql_model": SQL_MODEL_NAME,
            "analysis_model": ANALYSIS_MODEL_NAME,
            "embedding_model": self.retriever.embedding_model_name,
            "models_loaded": self.are_models_loaded(),
        }

    def get_schema_info(self) -> Dict[str, Any]:
        """Get information about the available schema tables."""
        return {
            "total_tables": len(self.table_chunks),
            "tables": [chunk["table"] for chunk in self.table_chunks],
            "embeddings_created": os.path.exists(os.path.join(EMBEDDINGS_PATH, "table_embeddings.npy"))
        }