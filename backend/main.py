# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os

from backend.analysis import compute_summary
from backend.llm_agent import answer_question


# -----------------------------------------------------------------------------
# FastAPI app (explicit docs so /docs, /redoc, /openapi.json always work)
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Career Insights API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# -----------------------------------------------------------------------------
# CORS (add your exact Vercel URL below)
# -----------------------------------------------------------------------------
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://vercel.com/nayeems-projects-1fcf4b20/career-insights",
    # Add your deployed frontend (Vercel) below, e.g.:
    # "https://career-insights-sand.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app$",  # allow Vercel previews too
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Small health + root endpoints so Render checks never 404
# -----------------------------------------------------------------------------
@app.get("/", tags=["system"])
def root():
    return {"status": "ok", "service": "career-insights-api"}

@app.get("/healthz", tags=["system"])
def healthz():
    return {"status": "healthy"}

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
# Expect project structure:
#   backend/
#     main.py
#     analysis.py
#     llm_agent.py
#   data/
#     career_data.csv
#   frontend/
#     index.html
#
# We resolve the CSV path relative to this file.
# -----------------------------------------------------------------------------
DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "career_data.csv")
)

def _load_dataframe() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            # Normalize expected columns if needed (safe defaults)
            # Expected by analysis.py: columns like Employed/EmploymentStatus, Salary, Sector, SupportService, Program
            # Create missing columns if absent to avoid crashes
            for col in ["Employed", "EmploymentStatus", "Salary", "Sector", "SupportService", "Program"]:
                if col not in df.columns:
                    df[col] = pd.Series(dtype="object")
            return df
        except Exception as e:
            # If CSV is corrupt, return empty frame to keep API alive
            print(f"[WARN] Failed reading CSV: {e}")
            return pd.DataFrame(columns=["Employed", "EmploymentStatus", "Salary", "Sector", "SupportService", "Program"])
    else:
        print(f"[WARN] Data file not found at {DATA_PATH}. Serving empty dataset.")
        return pd.DataFrame(columns=["Employed", "EmploymentStatus", "Salary", "Sector", "SupportService", "Program"])

df: pd.DataFrame = _load_dataframe()

# Precompute summary at startup (fast)
try:
    summary_cache = compute_summary(df)
except Exception as e:
    print(f"[WARN] compute_summary failed: {e}")
    summary_cache = {"records": 0, "employment_rate": 0.0, "avg_salary": 0.0, "by_sector": {}, "support_usage": {}}

# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
class Question(BaseModel):
    question: str

# -----------------------------------------------------------------------------
# API routes
# -----------------------------------------------------------------------------
@app.get("/statistics", tags=["analytics"])
def get_statistics():
    """
    Returns cached analytics summary computed from the CSV.
    """
    return summary_cache

@app.post("/ask", tags=["qa"])
def ask_question(q: Question):
    """
    Answers natural-language questions using rule-based + (optional) Gemini.
    If GEMINI_API_KEY is not set or the call fails, llm_agent should gracefully
    fall back to deterministic heuristics.
    """
    try:
        ans = answer_question(q.question, df)
        return {"question": q.question, "answer": ans}
    except Exception as e:
        # Never break the API—return a clear message
        return {
            "question": q.question,
            "answer": f"Sorry, I couldn't process that right now. ({type(e).__name__}: {e})"
        }
