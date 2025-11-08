# Career Outcome & Post‑Graduation Support Insights

This project demonstrates an agentic AI solution for generating insights from university career outcome data and post‑graduation support services.  It uses **LangChain** to orchestrate queries over a structured dataset and a **FastAPI** backend to expose aggregated statistics and a question‑answer endpoint.  A simple HTML front‑end is included for demonstration purposes.

## Features

- **Data ingestion** from a CSV file containing career outcomes and support services.
- **Aggregated statistics** such as employment rate and median salary.
- **AI‑driven Q&A** using a Language Model via LangChain (requires an API key).  The default implementation falls back to simple heuristics if no key is provided.
- **REST API** implemented with FastAPI.
- **Simple web interface** for querying the API.

## Prerequisites

- Python 3.10+
- Node.js (optional) if you replace the front‑end with a React app.

### Python dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

This will install `fastapi`, `uvicorn`, `pandas`, `langchain`, `openai`, `pydantic`, and `chromadb`.

## Usage

1. Place your career outcomes data in `data/career_data.csv`.  The sample file included uses the columns: `StudentId`, `Program`, `GraduationYear`, `EmploymentStatus`, `Salary`, `Sector`, `SupportServicesUsed`.

2. Run the API:

```bash
uvicorn backend.main:app --reload
```

3. Open the simple front‑end:

Open `frontend/index.html` in your browser.  It contains forms to fetch statistics and ask natural language questions.

4. (Optional) Use an external Language Model:

Edit `backend/llm_agent.py` and set your OpenAI API key or configure another provider.  The default implementation uses a placeholder summarization based on aggregated statistics.