from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os

from backend.analysis import compute_summary
from backend.llm_agent import answer_question

# ✅ Explicitly turn on docs and set openapi path
app = FastAPI(
    title="Career Insights API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    # "https://<your-vercel-app>.vercel.app",  # add when frontend deploys
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Root & health endpoints so Render checks don’t 404
@app.get("/", tags=["system"])
def root():
    return {"status": "ok", "service": "career-insights-api"}

@app.get("/healthz", tags=["system"])
def healthz():
    return {"status": "healthy"}

# ---- data load
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "career_data.csv")
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    df = pd.DataFrame(columns=["StudentID","Program","Employed","Salary","Sector","SupportService"])

summary_cache = compute_summary(df)

# ---- schema
class Question(BaseModel):
    question: str

# ---- routes
@app.get("/statistics", tags=["analytics"])
async def get_statistics():
    return summary_cache

@app.post("/ask", tags=["qa"])
async def ask_question(q: Question):
    ans = answer_question(q.question, df)
    return {"question": q.question, "answer": ans}
