"""
SCRIPTBEES ASSISTANT - FINAL CLEAN VERSION
-----------------------------------------
✔ Uses ScriptBees FAISS documents only
✔ Always returns REAL ScriptBees URLs from pages_meta.json
✔ NO fake websites in answer
✔ NO voice output
✔ NO proxies (safe for Render)
✔ Stable OpenAI generation
"""

import os
import json
import time
import logging
import hashlib
from typing import List
from pathlib import Path

# ------------------------------
# Remove proxy variables (Render injects these)
# ------------------------------
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

import httpx
from openai import OpenAI

# Create safe HTTP client
safe_http_client = httpx.Client(
    timeout=40,
    verify=True,
)

# ------------------------------
# Load env
# ------------------------------
from dotenv import load_dotenv

def find_env():
    cur = Path(__file__).resolve().parent
    for _ in range(10):
        if (cur / ".env").exists():
            return cur / ".env"
        cur = cur.parent
    return None

env = find_env()
if env:
    load_dotenv(env)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY = os.getenv("RAG_API_KEY", "change-me")

if not OPENAI_API_KEY:
    raise Exception("Missing OPENAI_API_KEY in .env")

# ------------------------------
# Logging
# ------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(" scriptbees ")

# ------------------------------
# Config
# ------------------------------
CONTENT_DIR = "content"
MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 1
MAX_TOKENS = 150
TEMPERATURE = 0.2

INDEX_PATH = f"{CONTENT_DIR}/pages.faiss"
META_PATH = f"{CONTENT_DIR}/pages_meta.json"
PAGES_PATH = f"{CONTENT_DIR}/pages.json"

# ------------------------------
# FastAPI App
# ------------------------------
from fastapi import FastAPI, Depends, HTTPException, Security, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

app = FastAPI(title="ScriptBees Assistant — Final Version")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

# ------------------------------
# Models
# ------------------------------
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)

class Source(BaseModel):
    url: str
    title: str
    score: float

class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    retrieved: List[Source]
    cached: bool
    response_time_seconds: float

# ------------------------------
# API Key Security
# ------------------------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(req: Request, key: str = Security(api_key_header)):
    incoming = key or ""

    if not incoming:
        auth = req.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            incoming = auth.split(" ", 1)[1]

    if incoming != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return incoming

# ------------------------------
# Cache
# ------------------------------
cache = {}
def ckey(q): return hashlib.md5(q.lower().encode()).hexdigest()

# ------------------------------
# Startup: Load FAISS + Model
# ------------------------------
retriever = None
generator = None

@app.on_event("startup")
async def startup():
    global retriever, generator

    logger.info("🚀 Starting ScriptBees AI Assistant...")

    import faiss
    from sentence_transformers import SentenceTransformer

    # Retriever
    class Retriever:
        def __init__(self):
            logger.info("📦 Loading FAISS + metadata...")

            self.model = SentenceTransformer(MODEL_NAME)
            self.index = faiss.read_index(INDEX_PATH)

            with open(META_PATH, "r") as f:
                self.meta = json.load(f)

            with open(PAGES_PATH, "r") as f:
                pages = json.load(f)

            self.pages = {p["id"]: p for p in pages}

            logger.info(f"✓ Loaded {self.index.ntotal} ScriptBees pages")

        def retrieve(self, question):
            vec = self.model.encode([question], normalize_embeddings=True).astype("float32")
            scores, idxs = self.index.search(vec, TOP_K)

            results = []
            for s, idx in zip(scores[0], idxs[0]):
                if idx == -1:
                    continue
                meta = self.meta[idx]
                page = self.pages.get(meta["id"], {})

                # Only ScriptBees pages (from your index)
                results.append({
                    "url": meta.get("url", ""),     # REAL ScriptBees URL
                    "title": meta.get("title", ""),
                    "score": float(s),
                    "text": page.get("text", "")[:1200]
                })
            return results

    # Generator
    class LLMGenerator:
        def __init__(self):
            self.client = OpenAI(
                api_key=OPENAI_API_KEY,
                http_client=safe_http_client
            )

        def generate(self, question, docs):
            context = docs[0]["text"]

            prompt = f"""
You are ScriptBees AI Assistant.

Answer ONLY using this ScriptBees content:

{context}

Question: {question}

Give a short and correct answer based ONLY on ScriptBees website.
"""

            res = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )
            return res.choices[0].message.content.strip()

    retriever = Retriever()
    generator = LLMGenerator()

    logger.info("✅ ScriptBees Assistant is READY")

# ------------------------------
# API Route
# ------------------------------
@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest, key: str = Depends(verify_api_key)):
    start = time.time()

    # Check cache
    if ckey(req.question) in cache:
        r = cache[ckey(req.question)]
        r["cached"] = True
        r["response_time_seconds"] = time.time() - start
        return r

    docs = retriever.retrieve(req.question)
    if not docs:
        return AskResponse(
            answer="No matching information found on ScriptBees.",
            sources=[],
            retrieved=[],
            cached=False,
            response_time_seconds=time.time() - start
        )

    answer = generator.generate(req.question, docs)

    resp = {
        "answer": answer,
        "sources": [docs[0]["url"]],       # REAL ScriptBees URL
        "retrieved": [Source(**docs[0])],
        "cached": False,
        "response_time_seconds": time.time() - start
    }

    cache[ckey(req.question)] = resp
    return resp

@app.get("/")
async def home():
    return {"status": "online", "bot": "ScriptBees AI"}

# ------------------------------
# Run Local
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
