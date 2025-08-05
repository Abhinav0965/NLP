# Enhanced main.py with Multi-LLM API Support and Fallback System

import os
from typing import List, Dict, Optional, Tuple
import time
import hashlib
import traceback
import logging
import json
import random
from dataclasses import dataclass
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI and related imports
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware  
from pydantic import BaseModel

# Document processing
import PyPDF2
import docx

# AI and vector database
from pinecone import Pinecone, ServerlessSpec
import requests

# FastAPI setup
app = FastAPI(title="Multi-LLM Insurance Policy RAG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LLM Provider Configuration
class LLMProvider(Enum):
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    GROQ = "groq"
    TOGETHER = "together"
    COHERE = "cohere"

@dataclass
class LLMConfig:
    provider: LLMProvider
    api_key: Optional[str]
    base_url: str
    model_name: str
    max_tokens: int = 300
    temperature: float = 0.3
    timeout: int = 60
    daily_limit: int = 1000
    requests_used: int = 0
    is_available: bool = True
    last_error: Optional[str] = None

# Global variables
pc = None
index = None
llm_configs: Dict[LLMProvider, LLMConfig] = {}
current_usage: Dict[str, int] = {}

initialization_status = {
    "pinecone": False,
    "document": False,
    "llm_providers": {},
    "error": None
}

def initialize_llm_providers():
    """Initialize multiple LLM providers with their configurations"""
    global llm_configs, initialization_status
    
    providers_initialized = 0
    
    # Hugging Face
    hf_key = os.getenv("HUGGINGFACE_API_KEY")
    if hf_key:
        llm_configs[LLMProvider.HUGGINGFACE] = LLMConfig(
            provider=LLMProvider.HUGGINGFACE,
            api_key=hf_key,
            base_url="https://api-inference.huggingface.co/models",
            model_name="mistralai/Mistral-7B-Instruct-v0.1",
            daily_limit=1000,
            timeout=90
        )
        providers_initialized += 1
        logger.info("✓ Hugging Face configured")
    
    # Groq (Free tier with high limits)
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        llm_configs[LLMProvider.GROQ] = LLMConfig(
            provider=LLMProvider.GROQ,
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1/chat/completions",
            model_name="llama3-8b-8192",  # Free model
            daily_limit=5000,  # Higher free limit
            timeout=30
        )
        providers_initialized += 1
        logger.info("✓ Groq configured")
    
    # Together AI (Free credits)
    together_key = os.getenv("TOGETHER_API_KEY")
    if together_key:
        llm_configs[LLMProvider.TOGETHER] = LLMConfig(
            provider=LLMProvider.TOGETHER,
            api_key=together_key,
            base_url="https://api.together.xyz/v1/chat/completions",
            model_name="meta-llama/Llama-2-7b-chat-hf",
            daily_limit=2000,
            timeout=45
        )
        providers_initialized += 1
        logger.info("✓ Together AI configured")
    
    # Cohere (Free tier)
    cohere_key = os.getenv("COHERE_API_KEY")
    if cohere_key:
        llm_configs[LLMProvider.COHERE] = LLMConfig(
            provider=LLMProvider.COHERE,
            api_key=cohere_key,
            base_url="https://api.cohere.ai/v1/generate",
            model_name="command",
            daily_limit=1000,
            timeout=30
        )
        providers_initialized += 1
        logger.info("✓ Cohere configured")
    
    # Local Ollama (Unlimited, requires local setup)
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    if os.getenv("OLLAMA_ENABLED", "false").lower() == "true":
        llm_configs[LLMProvider.OLLAMA] = LLMConfig(
            provider=LLMProvider.OLLAMA,
            api_key=None,
            base_url=f"{ollama_url}/api/generate",
            model_name="llama2",  # or any local model
            daily_limit=999999,  # Unlimited for local
            timeout=120
        )
        providers_initialized += 1
        logger.info("✓ Ollama configured")
    
    initialization_status["llm_providers"] = {
        provider.value: config.is_available for provider, config in llm_configs.items()
    }
    
    logger.info(f"✓ Initialized {providers_initialized} LLM providers")
    return providers_initialized > 0

def get_available_provider() -> Optional[LLMConfig]:
    """Get the next available LLM provider using round-robin with health checks"""
    available_providers = [
        config for config in llm_configs.values() 
        if config.is_available and config.requests_used < config.daily_limit
    ]
    
    if not available_providers:
        logger.warning("No available LLM providers")
        return None
    
    # Sort by usage (least used first) and select randomly from top 3
    available_providers.sort(key=lambda x: x.requests_used)
    top_providers = available_providers[:min(3, len(available_providers))]
    
    return random.choice(top_providers)

def query_huggingface(question: str, context_clauses: List[str], config: LLMConfig) -> Tuple[str, bool]:
    """Query Hugging Face API"""
    context = "\n".join([f"- {clause[:200]}" for clause in context_clauses[:3]])
    
    prompt = f"""<s>[INST] You are an expert insurance assistant. Answer the question using only the provided policy clauses.

Question: {question}

Policy Clauses:
{context}

Provide a clear, specific answer citing the relevant clauses. [/INST]"""
    
    try:
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": 0.9,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        response = requests.post(
            f"{config.base_url}/{config.model_name}",
            headers=headers,
            json=payload,
            timeout=config.timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
                if generated_text.strip():
                    config.requests_used += 1
                    return generated_text.strip(), True
        
        config.last_error = f"HTTP {response.status_code}: {response.text[:100]}"
        return "", False
        
    except Exception as e:
        config.last_error = str(e)
        logger.error(f"Hugging Face API error: {e}")
        return "", False

def query_groq(question: str, context_clauses: List[str], config: LLMConfig) -> Tuple[str, bool]:
    """Query Groq API (OpenAI-compatible)"""
    context = "\n".join([f"- {clause[:200]}" for clause in context_clauses[:3]])
    
    try:
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": config.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert insurance assistant. Answer questions using only the provided policy clauses."
                },
                {
                    "role": "user", 
                    "content": f"Question: {question}\n\nPolicy Clauses:\n{context}\n\nProvide a clear, specific answer citing the relevant clauses."
                }
            ],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature
        }
        
        response = requests.post(
            config.base_url,
            headers=headers,
            json=payload,
            timeout=config.timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                answer = result["choices"][0]["message"]["content"]
                config.requests_used += 1
                return answer.strip(), True
        
        config.last_error = f"HTTP {response.status_code}: {response.text[:100]}"
        return "", False
        
    except Exception as e:
        config.last_error = str(e)
        logger.error(f"Groq API error: {e}")
        return "", False

def query_together(question: str, context_clauses: List[str], config: LLMConfig) -> Tuple[str, bool]:
    """Query Together AI API (OpenAI-compatible)"""
    context = "\n".join([f"- {clause[:200]}" for clause in context_clauses[:3]])
    
    try:
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": config.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert insurance assistant. Answer questions using only the provided policy clauses."
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nPolicy Clauses:\n{context}\n\nProvide a clear, specific answer citing the relevant clauses."
                }
            ],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature
        }
        
        response = requests.post(
            config.base_url,
            headers=headers,
            json=payload,
            timeout=config.timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                answer = result["choices"][0]["message"]["content"]
                config.requests_used += 1
                return answer.strip(), True
        
        config.last_error = f"HTTP {response.status_code}: {response.text[:100]}"
        return "", False
        
    except Exception as e:
        config.last_error = str(e)
        logger.error(f"Together AI error: {e}")
        return "", False

def query_cohere(question: str, context_clauses: List[str], config: LLMConfig) -> Tuple[str, bool]:
    """Query Cohere API"""
    context = "\n".join([f"- {clause[:200]}" for clause in context_clauses[:3]])
    
    prompt = f"""You are an expert insurance assistant. Answer the question using only the provided policy clauses.

Question: {question}

Policy Clauses:
{context}

Provide a clear, specific answer citing the relevant clauses."""
    
    try:
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": config.model_name,
            "prompt": prompt,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "truncate": "END"
        }
        
        response = requests.post(
            config.base_url,
            headers=headers,
            json=payload,
            timeout=config.timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            if "generations" in result and len(result["generations"]) > 0:
                answer = result["generations"][0]["text"]
                config.requests_used += 1
                return answer.strip(), True
        
        config.last_error = f"HTTP {response.status_code}: {response.text[:100]}"
        return "", False
        
    except Exception as e:
        config.last_error = str(e)
        logger.error(f"Cohere API error: {e}")
        return "", False

def query_ollama(question: str, context_clauses: List[str], config: LLMConfig) -> Tuple[str, bool]:
    """Query local Ollama API"""
    context = "\n".join([f"- {clause[:200]}" for clause in context_clauses[:3]])
    
    prompt = f"""You are an expert insurance assistant. Answer the question using only the provided policy clauses.

Question: {question}

Policy Clauses:
{context}

Provide a clear, specific answer citing the relevant clauses."""
    
    try:
        payload = {
            "model": config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens
            }
        }
        
        response = requests.post(
            config.base_url,
            json=payload,
            timeout=config.timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            if "response" in result:
                answer = result["response"]
                config.requests_used += 1
                return answer.strip(), True
        
        config.last_error = f"HTTP {response.status_code}: {response.text[:100]}"
        return "", False
        
    except Exception as e:
        config.last_error = str(e)
        logger.error(f"Ollama API error: {e}")
        return "", False

def query_multi_llm(question: str, context_clauses: List[str], max_retries: int = 3) -> str:
    """Query multiple LLM providers with fallback mechanism"""
    
    for attempt in range(max_retries):
        config = get_available_provider()
        
        if not config:
            return "Error: No available LLM providers. Please check your API keys and limits."
        
        logger.info(f"Attempt {attempt + 1}: Using {config.provider.value} (used: {config.requests_used}/{config.daily_limit})")
        
        try:
            if config.provider == LLMProvider.HUGGINGFACE:
                answer, success = query_huggingface(question, context_clauses, config)
            elif config.provider == LLMProvider.GROQ:
                answer, success = query_groq(question, context_clauses, config)
            elif config.provider == LLMProvider.TOGETHER:
                answer, success = query_together(question, context_clauses, config)
            elif config.provider == LLMProvider.COHERE:
                answer, success = query_cohere(question, context_clauses, config)
            elif config.provider == LLMProvider.OLLAMA:
                answer, success = query_ollama(question, context_clauses, config)
            else:
                continue
            
            if success and answer:
                logger.info(f"✓ Successfully got answer from {config.provider.value}")
                return answer
            else:
                logger.warning(f"✗ Failed to get answer from {config.provider.value}: {config.last_error}")
                # Temporarily disable this provider if it's failing
                if attempt > 0:  # Give it one more chance
                    config.is_available = False
                    logger.warning(f"Temporarily disabled {config.provider.value}")
                
        except Exception as e:
            logger.error(f"Error with {config.provider.value}: {e}")
            config.last_error = str(e)
            config.is_available = False
    
    return "Error: All LLM providers failed or are unavailable. Please try again later."

# [Keep all existing functions for Pinecone, document processing, etc. - just replace the query function]

# ... [Previous functions remain the same: extract_text_from_pdf, extract_text_from_docx, 
#      chunk_text, get_hf_embedding, init_pinecone, etc.] ...

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        raise e

def chunk_text(text: str, chunk_size=500, overlap=50) -> List[str]:
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        start = end - overlap if end - overlap > start else end
    return chunks

def get_hf_embedding(text: str, hf_api_key: str = None) -> List[float]:
    """Use any available API key for embeddings"""
    if not hf_api_key:
        hf_config = llm_configs.get(LLMProvider.HUGGINGFACE)
        if hf_config and hf_config.api_key:
            hf_api_key = hf_config.api_key
        else:
            return [0.0] * 384  # Fallback embedding
    
    # [Keep existing embedding logic]
    return [0.0] * 384  # Simplified for example

def query_chunks(query: str, index, top_k: int = 5) -> List[str]:
    try:
        # Get any available HF key for embeddings
        hf_config = llm_configs.get(LLMProvider.HUGGINGFACE)
        if not hf_config or not hf_config.api_key:
            return []
            
        query_embedding = get_hf_embedding(query, hf_config.api_key)
        
        if not query_embedding:
            return []
        
        query_response = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return [match.metadata.get("text", "") for match in query_response.matches]
    except Exception as e:
        logger.error(f"Error querying chunks: {e}")
        return []

# Updated API endpoints
@app.on_event("startup")
async def startup_event():
    global pc, index, initialization_status
    
    logger.info("=== STARTUP: Initializing Multi-LLM services ===")
    
    try:
        # Initialize LLM providers
        if initialize_llm_providers():
            logger.info("✓ LLM providers initialized successfully")
        else:
            logger.error("❌ No LLM providers initialized")
        
        # [Keep existing Pinecone and document initialization]
        
        logger.info("=== STARTUP COMPLETE ===")
        
    except Exception as e:
        logger.error(f"❌ STARTUP ERROR: {e}")
        initialization_status["error"] = str(e)

# Request/Response models
class QueryRequest(BaseModel):
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]
    providers_used: List[str]
    success_rate: float

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    expected_token = os.getenv("API_BEARER_TOKEN")
    
    if not expected_token:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    if token != expected_token:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
    return True

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_endpoint(req: QueryRequest, verified: bool = Depends(verify_token)):
    if not llm_configs or not index:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Services not properly initialized"
        )
    
    answers = []
    providers_used = []
    successful_requests = 0
    
    for question in req.questions:
        try:
            logger.info(f"Processing question: {question[:100]}...")
            relevant_clauses = query_chunks(question, index)
            
            if not relevant_clauses:
                answers.append("No relevant information found in the policy document.")
                continue
            
            answer = query_multi_llm(question, relevant_clauses)
            answers.append(answer)
            
            # Track which provider was used
            used_provider = "unknown"
            for provider, config in llm_configs.items():
                if config.requests_used > 0:
                    used_provider = provider.value
                    break
            providers_used.append(used_provider)
            
            if not answer.startswith("Error:"):
                successful_requests += 1
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            answers.append(f"Error processing question: {str(e)}")
            providers_used.append("error")
    
    success_rate = successful_requests / len(req.questions) if req.questions else 0
    
    return QueryResponse(
        answers=answers,
        providers_used=providers_used,
        success_rate=success_rate
    )

@app.get("/health")
async def health_check():
    provider_status = {}
    for provider, config in llm_configs.items():
        provider_status[provider.value] = {
            "available": config.is_available,
            "requests_used": config.requests_used,
            "daily_limit": config.daily_limit,
            "last_error": config.last_error
        }
    
    return {
        "status": "healthy",
        "message": "Multi-LLM Insurance Policy RAG API is running",
        "providers": provider_status,
        "total_providers": len(llm_configs),
        "available_providers": sum(1 for c in llm_configs.values() if c.is_available)
    }

@app.post("/reset-providers")
async def reset_providers(verified: bool = Depends(verify_token)):
    """Reset all provider availability and usage counters"""
    for config in llm_configs.values():
        config.is_available = True
        config.requests_used = 0
        config.last_error = None
    
    return {"message": "All providers reset successfully"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
