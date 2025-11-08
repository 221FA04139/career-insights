"""
Utility functions for computing statistics from the career outcomes dataset.
Robust to either:
  - Employed (0/1) column, or
  - EmploymentStatus ("Employed"/"Unemployed") column.
"""

from __future__ import annotations
import pandas as pd
from typing import Dict, Any


def _series_to_bool_employed(df: pd.DataFrame) -> pd.Series:
    """
    Returns a boolean Series 'is_employed' regardless of source schema.
      - If 'Employed' exists: treat 1/True as employed.
      - Else if 'EmploymentStatus' exists: treat 'Employed' (case-insensitive) as employed.
      - Else: all False.
    """
    if "Employed" in df.columns:
        # Coerce to int/bool, then True if == 1
        try:
            return (df["Employed"].astype("Int64").fillna(0) == 1)
        except Exception:
            return df["Employed"].astype(bool).fillna(False)

    if "EmploymentStatus" in df.columns:
        return df["EmploymentStatus"].astype(str).str.lower().eq("employed")

    return pd.Series([False] * len(df), index=df.index)


def _safe_median(series: pd.Series) -> float | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.median()) if len(s) else None


def compute_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute aggregated statistics from the dataset.

    Expected columns (any subset):
      - StudentID
      - Program
      - Employed (0/1) OR EmploymentStatus ("Employed"/"Unemployed")
      - Salary (numeric, INR per year)
      - Sector
      - SupportService
    """
    out: Dict[str, Any] = {}

    if df is None or df.empty:
        return {
            "count": 0,
            "employment_rate_pct": 0.0,
            "median_salary_inr": None,
            "by_program": [],
            "by_sector_counts": [],
            "top_support_services": [],
        }

    is_emp = _series_to_bool_employed(df)

    total = len(df)
    employed_count = int(is_emp.sum())
    employment_rate = (employed_count / total * 100.0) if total else 0.0

    median_salary = _safe_median(df.get("Salary"))

    # By Program: employment rate & median salary
    by_program = []
    if "Program" in df.columns:
        for prog, sub in df.groupby("Program"):
            sub_emp = _series_to_bool_employed(sub)
            rate = (float(sub_emp.mean()) * 100.0) if len(sub) else 0.0
            med_sal = _safe_median(sub.get("Salary"))
            by_program.append(
                {
                    "program": str(prog),
                    "count": int(len(sub)),
                    "employment_rate_pct": round(rate, 2),
                    "median_salary_inr": None if med_sal is None else int(med_sal),
                }
            )
        # Sort by employment rate desc
        by_program.sort(key=lambda r: r["employment_rate_pct"], reverse=True)

    # By Sector counts (only for employed rows)
    by_sector_counts = []
    if "Sector" in df.columns:
        sector_counts = df.loc[is_emp, "Sector"].fillna("").replace("", pd.NA).dropna().value_counts()
        for sector, cnt in sector_counts.items():
            by_sector_counts.append({"sector": str(sector), "count": int(cnt)})

    # Top support services
    top_support = []
    if "SupportService" in df.columns:
        svc_counts = df["SupportService"].fillna("").replace("", pd.NA).dropna().value_counts()
        for svc, cnt in svc_counts.head(10).items():
            top_support.append({"service": str(svc), "count": int(cnt)})

    out.update(
        {
            "count": total,
            "employed": employed_count,
            "employment_rate_pct": round(employment_rate, 2),
            "median_salary_inr": None if median_salary is None else int(median_salary),
            "by_program": by_program,
            "by_sector_counts": by_sector_counts,
            "top_support_services": top_support,
        }
    )
    return out
