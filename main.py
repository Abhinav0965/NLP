# main.py - Updated for Together AI (replacing Gemini)

import os
from typing import List
import time
import hashlib
import traceback
import logging
import requests

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import PyPDF2
import docx
from pinecone import Pinecone, ServerlessSpec

# FastAPI setup
app = FastAPI(title="Insurance Policy RAG API with Together AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals
together_api_key = None
pc = None
index = None
initialization_status = {"gemini": False, "pinecone": False, "document": False, "error": None}

# Initialize Together AI key
def initialize_services():
    global initialization_status
    try:
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY environment variable not set")

        initialization_status["gemini"] = True
        return api_key
    except Exception as e:
        initialization_status["error"] = str(e)
        raise e

# Initialize Pinecone
def init_pinecone():
    global pc, index, initialization_status
    try:
        if pc is None:
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("PINECONE_API_KEY not set")
            pc = Pinecone(api_key=api_key)

        index_name = "policy-docs-gemini-hash"
        if index is None:
            if index_name not in [i.name for i in pc.list_indexes()]:
                pc.create_index(index_name, 512, "cosine", ServerlessSpec("aws", "us-east-1"))
                for _ in range(60):
                    if pc.describe_index(index_name).status["ready"]:
                        break
                    time.sleep(2)
            index = pc.Index(index_name)
            initialization_status["pinecone"] = True

        return pc, index
    except Exception as e:
        initialization_status["error"] = str(e)
        raise e

# Extract text

def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return "\n".join([page.extract_text() or "" for page in reader.pages])
    elif ext == ".docx":
        return "\n".join([p.text for p in docx.Document(file_path).paragraphs])
    else:
        raise ValueError(f"Unsupported format: {ext}")

# Chunking

def chunk_text(text, chunk_size=500, overlap=50):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap if end - overlap > start else end
    return chunks

# Embedding

def get_simple_embedding(text: str) -> List[float]:
    text = text.lower().strip()
    embeddings = []
    for i in range(16):
        hash_obj = hashlib.md5(f"{text}_{i}".encode())
        for byte_val in hash_obj.digest():
            embeddings.append((byte_val / 255.0) * 2 - 1)
            if len(embeddings) >= 256:
                break
    word_features = [len(text)/1000, len(text.split())/100, 0.0]  # add more if needed
    word_features += [0.0] * (256 - len(word_features))
    return (embeddings + word_features)[:512]

# Together AI call

def query_together_llm(question: str, context: List[str], api_key: str) -> str:
    prompt = f"""You are an expert assistant who answers insurance policy questions precisely and cites the clauses.

Question: {question}

Use ONLY the following clauses:
{chr(10).join(['- ' + clause for clause in context])}

Answer:
"""
    try:
        res = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            }
        )
        return res.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Pinecone Ops

def upsert_chunks(chunks, index, api_key):
    vectors = []
    for i, chunk in enumerate(chunks):
        emb = get_simple_embedding(chunk)
        vectors.append({"id": f"chunk-{i}-{int(time.time())}", "values": emb, "metadata": {"text": chunk}})
    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i:i + 100])

def query_chunks(query, index, api_key, top_k=5):
    emb = get_simple_embedding(query)
    matches = index.query(vector=emb, top_k=top_k, include_metadata=True)
    return [m.metadata.get("text", "") for m in matches.matches]

# Document indexing

def initialize_policy_document():
    global index, together_api_key
    path = "policy.pdf"
    if not os.path.exists(path): return
    text = extract_text(path)
    chunks = chunk_text(text)
    try:
        if index.describe_index_stats().total_vector_count > 0:
            index.delete(delete_all=True)
        upsert_chunks(chunks, index, together_api_key)
        initialization_status["document"] = True
    except Exception as e:
        initialization_status["error"] = str(e)
        raise e

# Auth
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    expected = os.getenv("API_BEARER_TOKEN")
    if not expected or token != expected:
        raise HTTPException(status_code=403, detail="Invalid token")
    return True

# Models
class QueryRequest(BaseModel):
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

@app.on_event("startup")
async def startup():
    global together_api_key, pc, index
    try:
        together_api_key = initialize_services()
        pc, index = init_pinecone()
        initialize_policy_document()
    except Exception as e:
        traceback.print_exc()

@app.post("/hackrx/run", response_model=QueryResponse)
async def run(req: QueryRequest, verified: bool = Depends(verify_token)):
    if not together_api_key or not index:
        raise HTTPException(status_code=503, detail="Services not initialized")
    answers = []
    for q in req.questions:
        clauses = query_chunks(q, index, together_api_key)
        if not clauses:
            answers.append("No relevant info found.")
        else:
            answers.append(query_together_llm(q, clauses, together_api_key))
    return QueryResponse(answers=answers)

@app.get("/health")
async def health():
    return {"status": "healthy", "init": initialization_status}

@app.get("/")
async def root():
    return {"message": "Insurance Policy RAG API using Together AI"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
