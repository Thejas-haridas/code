import torch
import time
import gc
import asyncio
import json
import os
import logging
import numpy as np
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Dict, Any, List , Optional , Union
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer ,OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModel
import uuid
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone
from passlib.context import CryptContext
import socket
import sqlite3

# JWT settings
SECRET_KEY = "your-secret-key"  # Replace with a strong key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Alias for code compatibility
DATA_TO_CREATE_TOKEN = SECRET_KEY


# --- 1. Configuration ---
# Model Configuration
SQL_MODEL_NAME = "defog/llama-3-sqlcoder-8b"  # For SQL generation
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # For embeddings

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Schema Management and SQL Generator", version="1.0.0")
executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)
app.state.loop = asyncio.get_event_loop()

# Hardcoded T-SQL Rules for different database servers
DATABASE_RULES = {
    "sqlserver": """
T-SQL Rules and Requirements:
- Use proper T-SQL syntax only (no PostgreSQL, MySQL, or other SQL dialects)
- Use DATEPART(), YEAR(), MONTH(), DAY() for date functions instead of EXTRACT()
- Use ISNULL() instead of COALESCE() when possible
- Use TOP instead of LIMIT
- For pagination, use OFFSET...FETCH NEXT instead of LIMIT
- Use proper JOIN syntax with explicit INNER/LEFT/RIGHT/FULL OUTER
- Do NOT use the column alias in the GROUP BY or ORDER BY clauses.
- Instead, repeat the full expression used in the SELECT clause inside GROUP BY and ORDER BY
- For string operations, use LEN() instead of LENGTH(), CHARINDEX() instead of POSITION()
- Use GETDATE() for current datetime, not NOW()
- For conditional logic, prefer CASE WHEN over IIF() for compatibility
- Do NOT use NULLS FIRST/NULLS LAST in ORDER BY (not supported in T-SQL)
- Use proper table aliases and qualify column names where ambiguous
- For date formatting, use FORMAT() or CONVERT() functions
- Use appropriate data types: VARCHAR(MAX), NVARCHAR(MAX), DECIMAL, DATETIME2, etc.
- Date time format eg:2024-11-21 06:57:57.000
""",
    "snowflake": """
Snowflake SQL Rules and Requirements:
- Use Snowflake SQL syntax
- Use EXTRACT() for date functions
- Use COALESCE() for null handling
- Use LIMIT for row limiting
- Use proper JOIN syntax with explicit INNER/LEFT/RIGHT/FULL OUTER
- Column aliases can be used in GROUP BY and ORDER BY clauses
- Use LENGTH() for string length, POSITION() for string search
- Use CURRENT_TIMESTAMP() for current datetime
- Use IFF() for simple conditional logic, CASE WHEN for complex conditions
- NULLS FIRST/NULLS LAST supported in ORDER BY
- Use proper table aliases and qualify column names where ambiguous
- For date formatting, use TO_CHAR() or TO_DATE() functions
- Use appropriate data types: VARCHAR, NUMBER, TIMESTAMP_NTZ, etc.
""",
    "postgresql": """
PostgreSQL Rules and Requirements:
- Use PostgreSQL syntax
- Use EXTRACT() for date functions
- Use COALESCE() for null handling
- Use LIMIT for row limiting
- Use proper JOIN syntax with explicit INNER/LEFT/RIGHT/FULL OUTER
- Column aliases can be used in GROUP BY and ORDER BY clauses
- Use LENGTH() for string length, POSITION() for string search
- Use NOW() for current datetime
- Use CASE WHEN for conditional logic
- NULLS FIRST/NULLS LAST supported in ORDER BY
- Use proper table aliases and qualify column names where ambiguous
- For date formatting, use TO_CHAR() or TO_DATE() functions
- Use appropriate data types: VARCHAR, NUMERIC, TIMESTAMP, etc.
"""
}


# Modified SchemaRetriever class to support session directories
class SchemaRetriever:
    def __init__(self, table_chunks: List[Dict], embedding_model_name: str, session_dir: str):
        self.table_chunks = table_chunks
        self.embedding_model_name = embedding_model_name
        self.session_dir = session_dir
        self.embeddings_file = os.path.join(session_dir, "table_embeddings.npy")
        self.metadata_file = os.path.join(session_dir, "table_metadata.json")
        self.embeddings = None
        self.embedding_model = None
    
    def load_embedding_model(self):
        """Load the sentence transformer model for embeddings."""
        logger.info(f"🔄 Loading embedding model: {self.embedding_model_name}")
        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        logger.info("✅ Embedding model loaded successfully")
    
    def create_embeddings(self):
        """Create embeddings for all table chunks."""
        logger.info("🔄 Creating embeddings for table chunks...")
        if self.embedding_model is None:
            self.load_embedding_model()
        # Prepare texts for embedding
        texts_to_embed = []
        for chunk in self.table_chunks:
            # Combine multiple fields for richer embeddings
            combined_text = f"{chunk['text']} {chunk['description']} {' '.join(chunk['keywords'])}"
            texts_to_embed.append(combined_text)
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts_to_embed)
        # Save embeddings and metadata
        np.save(self.embeddings_file, embeddings)
        with open(self.metadata_file, 'w') as f:
            json.dump({
                "table_names": [chunk["table"] for chunk in self.table_chunks],
                "embedding_dim": embeddings.shape[1],
                "num_tables": len(self.table_chunks)
            }, f)
        self.embeddings = embeddings
        logger.info(f"✅ Created and saved embeddings for {len(self.table_chunks)} tables")
    
    def load_embeddings(self):
        """Load existing embeddings from file."""
        if os.path.exists(self.embeddings_file) and os.path.exists(self.metadata_file):
            self.embeddings = np.load(self.embeddings_file)
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            logger.info(f"✅ Loaded embeddings for {metadata['num_tables']} tables")
            return True
        return False
    
    def retrieve_relevant_tables(self, question: str, top_k: int = 3) -> List[Dict]:
        """Retrieve top-k most relevant tables for the given question."""
        if self.embeddings is None:
            if not self.load_embeddings():
                self.create_embeddings()
        if self.embedding_model is None:
            self.load_embedding_model()
        # Create embedding for the question
        question_embedding = self.embedding_model.encode([question])
        # Calculate cosine similarities
        similarities = cosine_similarity(question_embedding, self.embeddings)[0]
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        # Return relevant table chunks
        relevant_tables = [self.table_chunks[i] for i in top_indices]
        logger.info(f"Retrieved {len(relevant_tables)} relevant tables: {[t['table'] for t in relevant_tables]}")
        return relevant_tables
def construct_rag_prompt_with_rules(question: str, relevant_tables: List[Dict], join_conditions: str, database_rules: str) -> str:
    """Creates a structured prompt using retrieved schema elements and database-specific rules."""
    # Build schema section from retrieved tables
    schema_section = ""
    for table_info in relevant_tables:
        schema_section += f"""
TABLE: {table_info['table']}
Description: {table_info['description']}
Columns: {', '.join(table_info['columns'])}
"""
    
    prompt = f"""### Task
Generate a SQL query that answers the following question using only the provided relevant schema information.

### Retrieved Schema Information
{schema_section}

### Join Conditions
{join_conditions}

### Database Rules
{database_rules}

### Question
{question}

### Instructions
- Use ONLY the tables and columns provided in the retrieved schema above
- Write clean, efficient SQL code with appropriate WHERE clauses for performance
- Use meaningful table aliases
- Add comments for complex logic if needed
- Ensure all column references are valid according to the provided schema
- Return only the SQL query without explanations

### SQL Query
```sql
"""
    return prompt

def extract_sql_from_response(response: str) -> str:
    """Extracts SQL code from the model's response with improved T-SQL parsing."""
    # First try to find SQL in code blocks
    if "```sql" in response:
        start = response.find("```sql") + 6
        end = response.find("```", start)
        if end != -1:
            sql_query = response[start:end].strip()
            return clean_tsql_query(sql_query)
    # Try to find SQL after "T-SQL Query" or similar markers
    markers = ["### T-SQL Query", "T-SQL Query:", "Query:", "SELECT", "WITH", "INSERT", "UPDATE", "DELETE"]
    for marker in markers:
        if marker in response:
            start = response.find(marker)
            if marker in ["SELECT", "WITH", "INSERT", "UPDATE", "DELETE"]:
                # For SQL keywords, include them in the result
                sql_part = response[start:]
            else:
                # For other markers, skip the marker itself
                sql_part = response[start + len(marker):]
            # Extract until we hit a non-SQL line or end
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
                    # Stop if we've started collecting SQL and hit a non-SQL line
                    break
            if sql_lines:
                return clean_tsql_query('\n'.join(sql_lines))
    # Fallback: return the response as-is if no SQL structure found
    return clean_tsql_query(response.strip())

def clean_tsql_query(sql_query: str) -> str:
    """Clean and validate T-SQL query for common issues."""
    if not sql_query:
        return sql_query
    # Remove common non-T-SQL patterns and fix them
    lines = sql_query.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.rstrip()
        if not line.strip():
            cleaned_lines.append(line)
            continue
        # Skip comment lines but keep SQL comments
        if line.strip().startswith('#'):
            continue
        # Fix common non-T-SQL patterns
        line_upper = line.upper()
        # Replace LIMIT with TOP (basic pattern)
        if 'LIMIT ' in line_upper and 'SELECT' in line_upper:
            # This is a simple replacement - more complex logic might be needed
            line = line.replace('LIMIT ', '-- LIMIT converted to TOP: ')
        # Remove NULLS FIRST/LAST
        if 'NULLS FIRST' in line_upper or 'NULLS LAST' in line_upper:
            line = line.replace('NULLS FIRST', '').replace('NULLS LAST', '')
            line = line.replace('nulls first', '').replace('nulls last', '')
        cleaned_lines.append(line)
    result = '\n'.join(cleaned_lines).strip()
    # Final cleanup - remove trailing semicolons if multiple exist
    while result.endswith(';;'):
        result = result[:-1]
    return result

def generate_sql_with_rag(question: str, retriever: SchemaRetriever) -> Tuple[str, List[str], float]:
    """Generates SQL query using RAG approach with schema retrieval."""
    start_time = time.time()
    # Retrieve relevant tables
    relevant_tables = retriever.retrieve_relevant_tables(question, top_k=3)
    retrieved_table_names = [table['table'] for table in relevant_tables]
    # Construct prompt with retrieved schema
    prompt = construct_rag_prompt(question, relevant_tables)
    # Generate SQL
    response = generate_text_optimized(prompt, app.state.sql_model, app.state.sql_tokenizer, max_new_tokens=300)
    sql_query = extract_sql_from_response(response)
    generation_time = time.time() - start_time
    logger.info(f"SQL generated with RAG in {generation_time:.2f}s using tables: {retrieved_table_names}")
    return sql_query, retrieved_table_names, generation_time


# --- Config ---
SECRET_KEY = "your-super-secret-key-change-this-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# --- DB Path ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "users.db")

# --- DB Connection ---
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# --- Initialize DB ---
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            disabled BOOLEAN NOT NULL DEFAULT 0
        )
    ''')
    conn.commit()
    cursor.close()
    conn.close()

@app.on_event("startup")
def startup_event():
    init_db()

# --- Models ---
class User(BaseModel):
    username: str
    password: str
    disabled: bool = False

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# --- Security ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

# --- User operations ---
def get_user(username: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT username, password, disabled FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if row:
            return {"username": row["username"], "password": row["password"], "disabled": bool(row["disabled"])}
        return None
    except Exception as e:
        print(f"Error getting user: {e}")
        return None

def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user['password']):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# --- Auth Dependencies ---
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception

    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: dict = Depends(get_current_user)):
    if current_user.get("disabled"):
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# --- Routes ---
@app.post("/register")
async def register(user: User):
    existing_user = get_user(user.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Username already exists"
        )

    hashed_password = get_password_hash(user.password)

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, password, disabled) VALUES (?, ?, ?)",
            (user.username, hashed_password, user.disabled)
        )
        conn.commit()
        cursor.close()
        conn.close()

        return {"message": "User registered successfully"}
    except Exception as e:
        print(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, 
        expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer"}
    
#----utility endpoints-----
# Pydantic Models
class TableColumn(BaseModel):
    column_name: str
    column_description: str

class TableSchema(BaseModel):
    table_name: str
    description: str
    columns: List[TableColumn]
    keywords: List[str]

class SchemaSetupRequest(BaseModel):
    database_type: str  # sqlserver, snowflake, postgresql
    tables: List[TableSchema]
    join_conditions: Optional[str] = ""

class SchemaSetupResponse(BaseModel):
    success: bool
    session_id: str
    message: str
    tables_processed: int
    embeddings_created: bool

class SQLGenerationRequest(BaseModel):
    session_id: str
    question: str

class SQLGenerationResponse(BaseModel):
    success: bool
    question: str
    retrieved_tables: List[str]
    generated_sql: str
    retrieval_time: float
    sql_generation_time: float
    
def create_table_chunks_from_input(tables: List[TableSchema]) -> List[Dict]:
    """Convert input table schemas to TABLE_CHUNKS format"""
    table_chunks = []
    
    for table in tables:
        # Create column list and column descriptions dict
        columns = [col.column_name for col in table.columns]
        
        # Create text representation
        text = f"Table: {table.table_name} | Columns: {', '.join(columns)}"
        
        table_chunk = {
            "type": "table",
            "table": table.table_name,
            "text": text,
            "description": table.description,
            "columns": columns,
            "keywords": table.keywords
        }
        table_chunks.append(table_chunk)
    
    return table_chunks

def create_table_descriptions_from_input(tables: List[TableSchema]) -> Dict:
    """Convert input table schemas to TABLE_DESCRIPTIONS format"""
    table_descriptions = {}
    
    for table in tables:
        columns_dict = {}
        for col in table.columns:
            columns_dict[col.column_name] = col.column_description
        
        table_descriptions[table.table_name] = {
            "description": table.description,
            "columns": columns_dict
        }
    
    return table_descriptions

def create_session_directory(session_id: str) -> str:
    """Create session directory for storing embeddings"""
    session_dir = os.path.join("sessions", session_id)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir

@app.post("/setup-schema", response_model=SchemaSetupResponse)
async def setup_schema(request: SchemaSetupRequest,current_user: dict = Depends(get_current_active_user)):
    """
    First API: Setup schema with tables, descriptions, columns, keywords and join conditions
    Creates embeddings and stores them in session-specific folder
    """
    try:
        # Generate UUID for session
        session_id = str(uuid.uuid4())
        
        # Validate database type
        if request.database_type not in DATABASE_RULES:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported database type. Supported types: {list(DATABASE_RULES.keys())}"
            )
        
        # Create session directory
        session_dir = create_session_directory(session_id)
        
        # Convert input to required formats
        table_chunks = create_table_chunks_from_input(request.tables)
        table_descriptions = create_table_descriptions_from_input(request.tables)
        
        # Save schema information
        schema_info = {
            "database_type": request.database_type,
            "join_conditions": request.join_conditions,
            "table_chunks": table_chunks,
            "table_descriptions": table_descriptions,
            "database_rules": DATABASE_RULES[request.database_type]
        }
        
        schema_file = os.path.join(session_dir, "schema_info.json")
        with open(schema_file, 'w') as f:
            json.dump(schema_info, f, indent=2)
        
        # Create SchemaRetriever instance with session-specific path
        retriever = SchemaRetriever(
            table_chunks=table_chunks,
            embedding_model_name="all-MiniLM-L6-v2",  # Default embedding model
            session_dir=session_dir
        )
        
        # Create embeddings
        retriever.create_embeddings()
        
        return SchemaSetupResponse(
            success=True,
            session_id=session_id,
            message="Schema setup completed successfully",
            tables_processed=len(request.tables),
            embeddings_created=True
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Schema setup failed: {str(e)}")

@app.post("/generate-sql", response_model=SQLGenerationResponse)
async def generate_sql(request: SQLGenerationRequest):
    """
    Second API: Generate SQL query from user question using stored schema and embeddings
    """
    try:
        # Check if session exists
        session_dir = os.path.join("sessions", request.session_id)
        if not os.path.exists(session_dir):
            raise HTTPException(status_code=404, detail=f"Session {request.session_id} not found")
        
        # Load schema information
        schema_file = os.path.join(session_dir, "schema_info.json")
        if not os.path.exists(schema_file):
            raise HTTPException(status_code=404, detail=f"Schema information not found for session {request.session_id}")
        
        with open(schema_file, 'r') as f:
            schema_info = json.load(f)
        
        # Create SchemaRetriever instance
        retriever = SchemaRetriever(
            table_chunks=schema_info["table_chunks"],
            embedding_model_name=EMBEDDING_MODEL_NAME,
            session_dir=session_dir
        )
        
        # Load existing embeddings
        if not retriever.load_embeddings():
            raise HTTPException(status_code=500, detail="Failed to load embeddings")
        
        # Generate SQL using existing function
        sql_query, retrieved_table_names, sql_generation_time = await app.state.loop.run_in_executor(
            executor, generate_sql_with_rag_session, 
            request.question, retriever, schema_info, request.top_k
        )
        
        if not sql_query:
            raise HTTPException(status_code=400, detail="Failed to generate SQL query")
        
        return SQLGenerationResponse(
            success=True,
            question=request.question,
            retrieved_tables=retrieved_table_names,
            generated_sql=sql_query,
            retrieval_time=0.0,  # Will be calculated in generate_sql_with_rag_session
            sql_generation_time=sql_generation_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SQL generation failed: {str(e)}")

        


if __name__ == "__main__":
   import uvicorn
   uvicorn.run(app, host="0.0.0.0", port=8000)

