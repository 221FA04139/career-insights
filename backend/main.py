"""
Entry point for the FastAPI application.

Exposes endpoints:
- GET  /statistics   → returns aggregated career outcome statistics
- POST /ask          → accepts a natural language question and returns an answer
- GET  /docs         → Swagger UI
"""

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os

from backend.analysis import compute_summary
from backend.llm_agent import answer_question

# ----------------------------
# App setup
# ----------------------------
app = FastAPI()

# Allow frontend (localhost:5173 or 127.0.0.1:5173) to call API
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Load dataset
# ----------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "career_data.csv")

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    df = pd.DataFrame(columns=["StudentID", "Program", "Employed", "Salary", "Sector", "SupportService"])

summary_cache = compute_summary(df)

# ----------------------------
# API Schemas
# ----------------------------
class Question(BaseModel):
    question: str

# ----------------------------
# Routes
# ----------------------------
@app.get("/statistics")
async def get_statistics():
    """
    Return pre-computed summary statistics from the dataset.
    """
    return summary_cache


@app.post("/ask")
async def ask_question(q: Question):
    """
    Answer a natural language question using the dataset + agent.
    """
    answer = answer_question(q.question, df)
    return {"question": q.question, "answer": answer}
