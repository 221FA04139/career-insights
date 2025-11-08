"""
Hybrid Q&A agent over the career dataset.

- Tries Gemini first (auto-discovers a supported model).
- Falls back to a simple rule-based responder if Gemini isn't available.
- You can force a model via env var:  GEMINI_MODEL=gemini-1.5-pro-latest
"""

from __future__ import annotations
import os
import pandas as pd

_USE_GEMINI = False
_MODEL = None
_listed_models = []

def _employment_rate(df: pd.DataFrame) -> float:
    if "Employed" in df.columns and len(df) > 0:
        return float((df["Employed"].astype("Int64").fillna(0) == 1).mean() * 100.0)
    if "EmploymentStatus" in df.columns and len(df) > 0:
        return float(df["EmploymentStatus"].astype(str).str.lower().eq("employed").mean() * 100.0)
    return 0.0

def _median_salary(df: pd.DataFrame):
    if "Salary" in df.columns:
        s = pd.to_numeric(df["Salary"], errors="coerce").dropna()
        if len(s): return float(s.median())
    return None

def _top_sector(df: pd.DataFrame):
    if "Sector" in df.columns and len(df):
        s = df["Sector"].astype(str).replace("", pd.NA).dropna()
        if len(s): return str(s.mode().iloc[0])
    return None

def _rule_based_answer(question: str, df: pd.DataFrame) -> str:
    q = (question or "").lower().strip()
    if any(k in q for k in ["employ", "placement", "placed"]):
        return f"Estimated employment rate is {_employment_rate(df):.1f}% (n={len(df)} records)."
    if any(k in q for k in ["salary", "package", "ctc"]):
        med = _median_salary(df)
        return f"The median salary is about ₹{med:,.0f}." if med is not None else "I don't have salary data available."
    if any(k in q for k in ["sector", "industry", "field"]):
        top = _top_sector(df)
        return f"The most common hiring sector appears to be {top}." if top else "I don't have sector data available."
    return ("Try asking: 'What is the employment rate?', "
            "'What is the median salary?', or 'Which sector hires most graduates?'")

# ---------- Gemini setup with auto model discovery ----------
try:
    import google.generativeai as genai
    _API_KEY = os.getenv("GEMINI_API_KEY")
    if _API_KEY:
        genai.configure(api_key=_API_KEY)

        def _pick_model() -> str:
            # Allow manual override
            forced = os.getenv("GEMINI_MODEL")
            if forced:
                return forced

            # List available models and pick one that supports generateContent
            models = list(genai.list_models())
            # keep for debugging if needed
            global _listed_models
            _listed_models = [m.name for m in models]

            def supports(m, method="generateContent"):
                try:
                    return method in getattr(m, "supported_generation_methods", [])
                except Exception:
                    return False

            # preference order
            preferred = [
                "gemini-1.5-flash-latest",
                "gemini-1.5-flash",
                "gemini-1.5-pro-latest",
                "gemini-1.5-pro",
                "gemini-pro",  # older
            ]
            names = {m.name for m in models if supports(m)}
            for p in preferred:
                if p in names:
                    return p
            # fallback: any model that supports generateContent
            if names:
                return sorted(names)[0]
            raise RuntimeError("No Gemini models supporting generateContent are available for this key/account.")

        model_name = _pick_model()
        _MODEL = genai.GenerativeModel(model_name)
        _USE_GEMINI = True
    else:
        _USE_GEMINI = False
except Exception:
    _USE_GEMINI = False

# ---------- Main answer function ----------
def answer_question(question: str, df: pd.DataFrame) -> str:
    if not _USE_GEMINI or _MODEL is None:
        return _rule_based_answer(question, df)

    try:
        parts = [f"Total records: {len(df)}",
                 f"Employment rate: {_employment_rate(df):.1f}%"]
        med = _median_salary(df)
        if med is not None: parts.append(f"Median salary: ₹{med:,.0f}")
        top = _top_sector(df)
        if top: parts.append(f"Top hiring sector: {top}")
        summary = "\n".join(parts)

        prompt = f"""
You are an assistant answering parent-style questions about graduates' career outcomes.
Be concise, numerical, and clear.

Dataset summary:
{summary}

Question: {question}
"""
        resp = _MODEL.generate_content(prompt)
        txt = (getattr(resp, "text", None) or "").strip()
        return txt or _rule_based_answer(question, df)
    except Exception as e:
        # graceful fallback
        return _rule_based_answer(question, df)
