# rag_chat_pipeline.py

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1) Query Expansion
# =========================================================
def expand_chat_query(user_query: str, ticker: str, company_name: str = "") -> List[str]:
    """
    Expand a user question to improve retrieval recall.
    Supports both static company facts and recent/live-ish market questions.
    """
    q = (user_query or "").strip()
    q_lower = q.lower()

    expansions = [q]

    if ticker:
        expansions.append(f"{ticker} {q}")

    if company_name:
        expansions.append(f"{company_name} {q}")

    # static company profile intent
    if any(k in q_lower for k in [
        "ceo", "cfo", "founder", "management", "leader", "leadership",
        "headquarters", "hq", "employees", "website", "business", "company",
        "industry", "sector", "what does", "profile"
    ]):
        expansions.extend([
            f"{ticker} company profile",
            f"{ticker} management leadership",
            f"{ticker} ceo headquarters business summary",
            f"{company_name} company profile"
        ])

    # products / services
    if any(k in q_lower for k in ["product", "products", "service", "services"]):
        expansions.extend([
            f"{ticker} products services",
            f"{company_name} products services"
        ])

    # financial intent
    if any(k in q_lower for k in [
        "revenue", "income", "margin", "eps", "profit", "financial",
        "cash flow", "balance sheet", "net income", "gross profit"
    ]):
        expansions.extend([
            f"{ticker} financial statements",
            f"{ticker} fundamentals profitability",
            f"{company_name} revenue income margin"
        ])

    # live / recent / market intent
    if any(k in q_lower for k in [
        "price", "close", "open", "high", "low", "volume",
        "today", "current", "latest", "live", "real time", "right now"
    ]):
        expansions.extend([
            f"{ticker} latest market snapshot",
            f"{ticker} latest trading snapshot",
            f"{company_name} current market data"
        ])

    # recent-news intent
    if any(k in q_lower for k in [
        "news", "recent", "week", "headline", "catalyst", "why is", "why did", "moving"
    ]):
        expansions.extend([
            f"{ticker} latest news",
            f"{company_name} recent headlines",
            f"{ticker} catalyst outlook"
        ])

    # options / positioning intent
    if any(k in q_lower for k in ["options", "option", "oi", "open interest", "strike", "derivatives"]):
        expansions.extend([
            f"{ticker} options open interest",
            f"{company_name} options positioning"
        ])

    # risk / valuation intent
    if any(k in q_lower for k in ["risk", "valuation", "worth", "buy", "invest", "expensive", "cheap"]):
        expansions.extend([
            f"{ticker} risks fundamentals",
            f"{company_name} valuation outlook"
        ])

    seen = set()
    unique_queries = []
    for item in expansions:
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            unique_queries.append(item)

    return unique_queries


# =========================================================
# 2) Build retrievable chat knowledge base
# =========================================================
def build_chat_knowledge_base(
    ticker: str,
    fundamentals: Dict[str, Any],
    df_price: Optional[pd.DataFrame] = None,
    df_financials: Optional[pd.DataFrame] = None,
    options_snap: Optional[Dict[str, Any]] = None,
    news_items: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Turn available stock evidence into retrievable mini-documents.
    Includes both static company facts and recent/live-ish market evidence.
    """
    kb: List[Dict[str, Any]] = []
    symbol = str(ticker).upper()
    company_name = fundamentals.get("Company Name", symbol) if fundamentals else symbol
    today_str = datetime.utcnow().strftime("%Y-%m-%d")

    # -------------------------
    # A) Static company profile
    # -------------------------
    if fundamentals:
        static_profile_text = (
            f"Company profile for {company_name} ({symbol}). "
            f"CEO: {fundamentals.get('CEO', 'N/A')}. "
            f"CFO: {fundamentals.get('CFO', 'N/A')}. "
            f"Founder: {fundamentals.get('Founder', 'N/A')}. "
            f"Sector: {fundamentals.get('Sector', 'N/A')}. "
            f"Industry: {fundamentals.get('Industry', 'N/A')}. "
            f"Headquarters: {fundamentals.get('Headquarters', 'N/A')}. "
            f"Country: {fundamentals.get('Country', 'N/A')}. "
            f"Website: {fundamentals.get('Website', 'N/A')}. "
            f"Employees: {fundamentals.get('Full Time Employees', 'N/A')}. "
            f"Exchange: {fundamentals.get('Exchange', 'N/A')}. "
            f"Currency: {fundamentals.get('Currency', 'N/A')}. "
            f"Business summary: {fundamentals.get('Business Summary', 'N/A')}"
        )

        kb.append({
            "id": f"{symbol}_profile_static_1",
            "text": static_profile_text,
            "meta": {
                "source": "Yahoo Finance Company Profile",
                "doc_type": "company_profile",
                "section": "Static Company Facts",
                "as_of_date": today_str,
                "url": fundamentals.get("Website", "")
            }
        })

        metrics = []
        for key in [
            "P/E Ratio (TTM)",
            "Revenue Growth (YoY)",
            "Profit Margin",
            "ROE (Return on Equity)",
            "Debt/Equity",
            "Market Cap",
            "Sector",
            "Industry",
            "Data Period"
        ]:
            if key in fundamentals:
                metrics.append(f"{key}: {fundamentals.get(key)}")

        if metrics:
            kb.append({
                "id": f"{symbol}_fundamentals_1",
                "text": f"Fundamental snapshot for {company_name} ({symbol}): " + " | ".join(metrics),
                "meta": {
                    "source": "Yahoo Finance Fundamentals",
                    "doc_type": "fundamentals",
                    "section": "Key Metrics",
                    "as_of_date": today_str,
                    "url": ""
                }
            })

    # -------------------------
    # B) Latest market snapshot
    # -------------------------
    if df_price is not None and not df_price.empty:
        try:
            latest = df_price.iloc[-1]

            open_px = float(latest["Open"]) if "Open" in latest and pd.notna(latest["Open"]) else None
            high_px = float(latest["High"]) if "High" in latest and pd.notna(latest["High"]) else None
            low_px = float(latest["Low"]) if "Low" in latest and pd.notna(latest["Low"]) else None
            close_px = float(latest["Close"]) if "Close" in latest and pd.notna(latest["Close"]) else None
            volume = float(latest["Volume"]) if "Volume" in latest and pd.notna(latest["Volume"]) else None

            price_date = today_str
            try:
                price_date = df_price.index[-1].strftime("%Y-%m-%d")
            except Exception:
                pass

            market_text = (
                f"{company_name} ({symbol}) latest market snapshot as of {price_date}. "
                f"Open={open_px if open_px is not None else 'N/A'}, "
                f"High={high_px if high_px is not None else 'N/A'}, "
                f"Low={low_px if low_px is not None else 'N/A'}, "
                f"Close={close_px if close_px is not None else 'N/A'}, "
                f"Volume={volume if volume is not None else 'N/A'}."
            )

            kb.append({
                "id": f"{symbol}_market_1",
                "text": market_text,
                "meta": {
                    "source": "Yahoo Finance Price History",
                    "doc_type": "market_data",
                    "section": "Latest Trading Snapshot",
                    "as_of_date": price_date,
                    "url": ""
                }
            })
        except Exception:
            pass

    # -------------------------
    # C) Financial statements
    # -------------------------
    if df_financials is not None and not df_financials.empty:
        for idx, row in df_financials.iterrows():
            year = row.get("Year", idx)
            metrics = []
            for col in df_financials.columns:
                if col == "Year":
                    continue
                metrics.append(f"{col}: {row.get(col)} million USD")

            kb.append({
                "id": f"{symbol}_financials_{year}",
                "text": f"{company_name} ({symbol}) financial statement for {year}: " + " | ".join(metrics),
                "meta": {
                    "source": "Yahoo Finance Financial Statements",
                    "doc_type": "financials",
                    "section": f"Annual Financials {year}",
                    "as_of_date": str(year),
                    "url": ""
                }
            })

    # -------------------------
    # D) Options snapshot
    # -------------------------
    if options_snap:
        kb.append({
            "id": f"{symbol}_options_1",
            "text": (
                f"{company_name} ({symbol}) options positioning. "
                f"Nearest expiration={options_snap.get('expiration', 'N/A')}; "
                f"Top call OI strike={options_snap.get('top_call_oi_strike', 'N/A')}; "
                f"Top put OI strike={options_snap.get('top_put_oi_strike', 'N/A')}; "
                f"Top call OI={options_snap.get('top_call_oi', 'N/A')}; "
                f"Top put OI={options_snap.get('top_put_oi', 'N/A')}."
            ),
            "meta": {
                "source": "Yahoo Finance Options",
                "doc_type": "options",
                "section": "Open Interest Snapshot",
                "as_of_date": today_str,
                "url": ""
            }
        })

    # -------------------------
    # E) News chunks
    # -------------------------
    for i, item in enumerate(news_items or [], start=1):
        kb.append({
            "id": f"{symbol}_news_{i}",
            "text": (
                f"News headline: {item.get('title', '')}. "
                f"Publisher: {item.get('publisher', 'Unknown')}. "
                f"Date: {item.get('date', 'N/A')}. "
                f"Tone: {item.get('tone_tag', 'Neutral')}. "
                f"Raw score: {item.get('raw_score', 0)}. "
                f"Sentiment weight: {item.get('sentiment_weight', 0.0):.2f}."
            ),
            "meta": {
                "source": item.get("publisher", "News"),
                "doc_type": "news",
                "section": f"Headline {i}",
                "as_of_date": item.get("date", "N/A"),
                "url": item.get("link", "")
            }
        })

    return kb


# =========================================================
# 3) Sparse retrieval
# =========================================================
def retrieve_chat_documents(
    user_query: str,
    ticker: str,
    company_name: str,
    knowledge_base: List[Dict[str, Any]],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k relevant documents using TF-IDF sparse retrieval.
    """
    if not knowledge_base:
        return []

    expanded_queries = expand_chat_query(
        user_query=user_query,
        ticker=ticker,
        company_name=company_name
    )

    texts = [doc["text"] for doc in knowledge_base]
    if not texts:
        return []

    vectorizer = TfidfVectorizer(stop_words="english")
    doc_matrix = vectorizer.fit_transform(texts)

    score_map = defaultdict(float)
    doc_lookup = {doc["id"]: doc for doc in knowledge_base}

    for q in expanded_queries:
        q_vec = vectorizer.transform([q])
        scores = cosine_similarity(q_vec, doc_matrix).ravel()

        for idx, score in enumerate(scores):
            doc_id = knowledge_base[idx]["id"]
            score_map[doc_id] = max(score_map[doc_id], float(score))

    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)

    results = []
    for doc_id, score in ranked[:top_k]:
        doc = dict(doc_lookup[doc_id])
        doc["retrieval_score"] = score
        results.append(doc)

    return results


# =========================================================
# 4) Context formatting
# =========================================================
def build_chat_context(retrieved_docs: List[Dict[str, Any]]) -> str:
    if not retrieved_docs:
        return "No retrieved documents."

    parts = []
    for i, doc in enumerate(retrieved_docs, start=1):
        meta = doc.get("meta", {})
        parts.append(
            f"[{i}] Source: {meta.get('source', 'Unknown Source')} | "
            f"Type: {meta.get('doc_type', 'N/A')} | "
            f"Section: {meta.get('section', 'N/A')} | "
            f"Date: {meta.get('as_of_date', 'N/A')}\n"
            f"Content: {doc.get('text', '')}"
        )

    return "\n\n---\n\n".join(parts)


def build_chat_references(retrieved_docs: List[Dict[str, Any]]) -> str:
    if not retrieved_docs:
        return "\n\n### References\nNone"

    lines = ["\n\n### References"]
    for i, doc in enumerate(retrieved_docs, start=1):
        meta = doc.get("meta", {})
        line = (
            f"[{i}] {meta.get('source', 'Unknown Source')} | "
            f"{meta.get('section', 'N/A')} | "
            f"{meta.get('as_of_date', 'N/A')}"
        )
        if meta.get("url"):
            line += f" | {meta.get('url')}"
        lines.append(line)

    return "\n".join(lines)


# =========================================================
# 5) Answer generation with citations
# =========================================================
def generate_chat_answer_with_citations(
    model,
    ticker: str,
    company_name: str,
    question: str,
    retrieved_docs: List[Dict[str, Any]],
    today: str
) -> str:
    """
    Generate grounded answer using ONLY retrieved evidence.
    """
    context = build_chat_context(retrieved_docs)
    references = build_chat_references(retrieved_docs)

    prompt = f"""
You are a professional investing assistant.

Today's date: {today}
Ticker: {ticker}
Company: {company_name}

Answer using ONLY the retrieved context below.

Rules:
1. Do not invent facts.
2. Every material claim must have inline citations like [1], [2].
3. If the evidence is insufficient, say exactly:
   "I don't have enough retrieved evidence to answer that confidently."
4. If the question is about recent or current market movement, make clear that the answer depends on the retrieved market/news context.
5. If the question asks for exact live market data beyond the retrieved context, clearly say that live confirmation may require external real-time APIs.
6. If sources support only part of the answer, answer only that supported part.
7. Be clear, concise, and professional.
8. Return markdown-friendly text.
9. After the answer, append the references section exactly once.
10. Do not cite sources that are not in the retrieved context.

User question:
{question}

Retrieved context:
{context}

Now write the final answer with inline citations.
After the answer, append the references section exactly once.
"""

    response = model.generate_content(prompt)
    text = getattr(response, "text", None) or "Error: Empty response"

    if "### References" not in text:
        text += references

    return text