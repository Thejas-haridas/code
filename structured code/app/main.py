from fastapi import FastAPI
from app.rag_with_faiss import SchemaRetriever

app = FastAPI(title="Structured Code API", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "Welcome to the Structured Code API"}

# Additional routes can be defined here as needed

# Initialize the SchemaRetriever or any other components if necessary
# app.state.retriever = SchemaRetriever(...)