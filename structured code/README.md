# Structured Code Project

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system that generates SQL queries based on user questions and analyzes the results. It utilizes FastAPI for the web framework and integrates various machine learning models for SQL generation and analysis.

## Project Structure
```
structured code
├── app
│   ├── main.py                # Entry point for the FastAPI application
│   ├── rag_with_faiss.py      # Main logic for the RAG system
│   ├── models
│   │   └── __init__.py        # Data models used in the application
│   ├── utils
│   │   └── __init__.py        # Utility functions for common tasks
│   └── schema
│       └── __init__.py        # Schema information and metadata
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Installation
To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd structured code
pip install -r requirements.txt
```

## Usage
To run the application, execute the following command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

You can then access the API at `http://localhost:8000`.

## API Endpoints
- **/generate-and-analyze-sql**: Generates SQL queries based on user questions and analyzes the results.
- **/health**: Checks the health status of the application.
- **/generate-sql-only**: Generates SQL queries without executing or analyzing them.
