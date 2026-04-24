"""
ai_advisor.py
─────────────
Optional AI-powered privacy risk summarizer.

Takes **only aggregated, anonymised metrics** — never raw PII —
and produces a short natural-language advisory paragraph.

Uses the Gemini API (via LangChain) if configured;
otherwise a deterministic template-based fallback is returned.
"""

from __future__ import annotations

import os
from typing import Any, Dict


def generate_ai_summary(metrics_dict: Dict[str, Any]) -> str:
    """
    Generate a short advisory paragraph from aggregated privacy metrics.

    Parameters
    ----------
    metrics_dict : dict
        Aggregated metrics with keys such as ``total_events``,
        ``high_risk_count``, ``cumulative_risk_score``, etc.
        **Must NOT contain any raw PII.**

    Returns
    -------
    str
        A human-readable advisory paragraph.
    """

    # ── Attempt AI-powered summary via Gemini (LangChain) ──────────────
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        try:
            return _call_gemini(api_key, metrics_dict)
        except Exception:
            # Silently fall back to template-based summary
            pass

    # ── Deterministic fallback ─────────────────────────────────────────
    return _template_summary(metrics_dict)


# ── Private helpers ────────────────────────────────────────────────────

def _call_gemini(api_key: str, metrics_dict: Dict[str, Any]) -> str:
    """
    Call the Gemini API via LangChain with a privacy-focused system prompt.

    Only sends aggregated numbers — no PII crosses the boundary.
    """
    import json
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
    except ImportError:
        return _template_summary(metrics_dict)

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.7,
        max_tokens=200
    )

    system_prompt = (
        "You are a student privacy advisor for a campus event system called LowKey Secure. "
        "Given ONLY aggregated, anonymised monthly metrics (never raw PII), "
        "write a concise 3-4 sentence advisory paragraph. "
        "Be empathetic, actionable, and clear. Do not use markdown formatting."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Here are the student's aggregated privacy metrics for the month:\n{metrics_json}\n\nProvide a brief privacy advisory.")
    ])

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({"metrics_json": json.dumps(metrics_dict, indent=2)})


def _template_summary(metrics_dict: Dict[str, Any]) -> str:
    """
    Deterministic template-based fallback — no external calls required.
    """
    total = metrics_dict.get("total_events", 0)
    high = metrics_dict.get("high_risk_count", 0)
    medium = metrics_dict.get("medium_risk_count", 0)
    score = metrics_dict.get("cumulative_risk_score", 0)
    velocity = metrics_dict.get("risk_velocity", 0.0)
    orgs = metrics_dict.get("unique_org_count", 0)
    repeated = metrics_dict.get("repeated_high_attr_count", 0)

    # No activity
    if total == 0:
        return (
            "You did not attend any events this month. "
            "Your privacy exposure is minimal. Stay cautious when sharing data in the future!"
        )

    parts: list[str] = []

    # Opening
    parts.append(
        f"This month you attended {total} event(s), sharing data with "
        f"{orgs} organiser(s)."
    )

    # Risk composition
    if high > 0:
        parts.append(
            f"Of these, {high} involved HIGH-risk data exposure — "
            f"be especially careful with sensitive attributes like phone or ID numbers."
        )
    elif medium > 0:
        parts.append(
            f"{medium} event(s) carried MEDIUM-risk exposure (name, email, etc.). "
            f"Consider limiting what you share."
        )
    else:
        parts.append(
            "All events were LOW-risk — great job keeping your data safe!"
        )

    # Repeated sensitive sharing
    if repeated > 0:
        parts.append(
            f"⚠️ {repeated} sensitive attribute(s) were shared with multiple organisers. "
            f"Try to avoid repeating sensitive data across events."
        )

    # Velocity
    if velocity > 10:
        parts.append(
            "Your risk exposure is increasing rapidly compared to last month. "
            "Consider pausing consent to high-risk events."
        )
    elif velocity < 0:
        parts.append(
            "Good news — your risk exposure decreased compared to last month. Keep it up!"
        )

    # Closing
    if score >= 30:
        parts.append(
            "Overall, your monthly risk score is HIGH. Review your consent choices carefully."
        )
    elif score >= 12:
        parts.append(
            "Your risk level is moderate — stay mindful of the events you share data with."
        )
    else:
        parts.append(
            "Your overall privacy posture is healthy. Keep reviewing event risk levels!"
        )

    return " ".join(parts)
