
# StockMaster – AI-Powered Stock Analysis Dashboard

## Overview
StockMaster is an AI-powered U.S. stock analysis dashboard built with **Python, Dash, and Plotly**.  
It integrates market data retrieval, financial statement analysis, technical indicators, valuation scenario simulation, news sentiment analysis, AI-generated investment commentary, and automated report export into a single workflow.

The system allows a user to input a stock ticker and automatically generate a **structured investment research view** similar to an institutional equity research report.

---

# Problem & Motivation

Modern stock research workflows are fragmented. Analysts typically must switch between multiple tools to complete a single analysis:

- market data platforms
- financial statement databases
- charting software
- news sources
- spreadsheet models
- document editors

This workflow is slow, manual, and inconsistent.

Key problems include:

1. **Fragmented financial data**
   - Market prices, fundamentals, and financial statements exist across multiple platforms.

2. **Disconnect between quantitative and qualitative analysis**
   - Traditional dashboards process numeric financial metrics but ignore narrative market signals such as news sentiment.

3. **Lack of automated research generation**
   - Most financial dashboards visualize data but do not produce structured research insights.

StockMaster addresses these issues by combining **data retrieval, analytics, and AI reasoning** into one integrated application.

---

# Solution Overview

The system architecture combines several layers:

### 1. Data Retrieval Layer
Market data is retrieved using:

- `yfinance` API for historical stock prices
- company fundamentals
- financial statements

### 2. Data Processing Layer
Financial data is processed using:

- `pandas` for data cleaning and transformation
- `pandas_ta` for technical indicator calculations

### 3. AI Analysis Layer
The system integrates **Google Gemini** to interpret financial signals and generate institutional-style investment commentary.

The AI analysis uses:

- company fundamentals
- simulated valuation scenarios
- news sentiment signals
- options market context

### 4. Interactive Dashboard
The frontend is built using:

- **Dash**
- **Plotly visualizations**
- **Bootstrap layout styling**

Users can explore financial data, technical indicators, and AI analysis interactively.

---

# Core Features

## 1. Stock Data Retrieval

The backend retrieves historical market data and fundamentals including:

- P/E Ratio (TTM)
- Revenue Growth
- Profit Margin
- ROE
- Debt / Equity
- Market Cap
- Sector
- Industry

This is implemented through the function:

```
get_stock_data()
```

---

## 2. Financial Statement Analysis

The system extracts key income statement data:

- Total Revenue
- Gross Profit
- Operating Income
- Net Income

Data is transformed into a **clean yearly P&L table (Million USD)** and visualized as a revenue vs. net income trend chart.

Function:

```
get_financial_statements()
```

---

## 3. Technical Analysis Dashboard

The dashboard visualizes stock price movements using:

- Candlestick chart
- Moving averages

Indicators displayed:

- MA(5)
- MA(20)
- MA(60)
- MA(90)
- MA(248)

These indicators help identify short‑term and long‑term market trends.

---

## 4. Valuation Scenario Simulation

The backend includes a rule-based valuation engine that evaluates the stock using:

- P/E ratio
- revenue growth
- profit margin
- market capitalization
- leverage
- sector / industry context
- news sentiment

The model produces three forward scenarios:

- **Bullish scenario**
- **Base scenario**
- **Bearish scenario**

Functions:

```
dynamic_valuation_model()
simulated_valuation()
```

---

## 5. News Sentiment Integration

Recent news headlines are collected and analyzed with a sentiment scoring system.

The sentiment logic:

1. Headlines are classified as positive / neutral / negative.
2. Sentiment weights are assigned.
3. Scores are aggregated.
4. Final sentiment is capped to avoid extreme distortion.

Functions:

```
get_recent_news()
summarize_news_sentiment()
```

This connects **market narratives** with valuation results.

---

## 6. AI Investment Plan Generation

The system uses **Google Gemini** to generate a structured investment analysis.

The prompt integrates:

- company fundamentals
- scenario valuation anchors
- recent news context
- options snapshot
- sentiment explanation

Function:

```
get_ai_investment_plan()
```

The output resembles an **institutional investment memo**.

---

## 7. AI Stock Chatbot

Users can ask follow‑up questions about the selected stock using a built‑in chatbot.

Example questions:

- “What are the key risks for this company?”
- “Explain the growth drivers.”
- “How does sentiment affect the price scenarios?”

This provides quick contextual Q&A on the selected ticker.

---

## 8. DOCX Report Export

The dashboard can export a **professional investment report** in `.docx` format.

The exported report includes:

- company fundamentals
- AI-generated investment commentary
- structured report sections
- disclaimer

Function:

```
generate_docx_report()
```

---

# Project Structure

```
.
├── stock_master_app_fixed.py
├── stock_data_utils_fixed.py
├── run_dashboard.bat
└── README.md
```

### stock_master_app_fixed.py

Main Dash application.

Responsibilities:

- dashboard layout
- charts and tables
- callbacks
- chatbot integration
- AI report display
- DOCX export

---

### stock_data_utils_fixed.py

Backend analytics module.

Responsibilities:

- market data retrieval
- financial statement parsing
- valuation simulation
- news sentiment scoring
- AI investment analysis
- DOCX report generation

---

### run_dashboard.bat

Windows launcher script.

Typical use:

- install dependencies
- start the Dash server
- open the dashboard locally

---

# Tech Stack

## Frontend
- Dash
- Plotly
- Bootstrap

## Backend
- Python
- pandas
- yfinance
- pandas_ta
- requests

## AI / NLP
- Google Gemini
- Prompt-based reasoning
- TF-IDF similarity ranking

## Export
- python-docx

---

# Workflow

1. User enters a stock ticker.
2. Market data and fundamentals are retrieved.
3. Technical indicators are calculated.
4. Financial statements are visualized.
5. Valuation scenarios are simulated.
6. Recent news is analyzed for sentiment.
7. Gemini generates an investment plan.
8. User can ask follow-up questions.
9. Report can be exported to DOCX.

---

# Installation

Install required packages:

```bash
pip install dash plotly pandas yfinance pandas-ta requests google-generativeai python-docx scikit-learn
```

---

# Environment Variables

Required:

```
GEMINI_API_KEY=your_api_key
NEWS_API_KEY=your_news_api_key
```

If the Gemini key is missing, AI analysis will not run.

---

# Running the Application

### Option 1

```
python stock_master_app_fixed.py
```

### Option 2

```
run_dashboard.bat
```

Then open:

```
http://127.0.0.1:8050/
```

---

# Current Limitations

- News sentiment is rule-based.
- Valuation engine is scenario-based rather than full financial modeling.
- RAG retrieval is lightweight (TF-IDF) instead of vector database.

---

# Future Improvements

Potential upgrades:

- vector database RAG system
- advanced news sentiment models
- portfolio analytics
- multi-stock comparison
- deeper options analysis
- richer DOCX export with charts

---

# Disclaimer

This project is for research and educational purposes only and does not constitute investment advice.
