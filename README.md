# StockMaster

### AI-Powered Stock Research & Investment Analysis Platform

StockMaster is an AI-powered equity research dashboard that integrates
financial data analysis, news sentiment intelligence, valuation
simulation, and Retrieval-Augmented Generation (RAG) AI reasoning into a
single application.

The system allows users to generate institutional-style investment
research views directly from a stock ticker using automated data
pipelines and AI analysis.

------------------------------------------------------------------------

# Problem & Motivation

Modern stock research workflows are fragmented. Analysts often rely on
multiple tools to complete a single analysis:

-   financial databases
-   charting platforms
-   news terminals
-   spreadsheet models
-   document editors

This leads to several issues:

1.  Fragmented financial data across platforms
2.  Manual research synthesis
3.  Lack of automated insights
4.  Weak connection between quantitative metrics and qualitative signals
    such as news sentiment

StockMaster integrates data retrieval, analytics, and AI reasoning into
a unified workflow.

------------------------------------------------------------------------

# System Architecture

User Input (Ticker / Question) ↓ Data Retrieval Layer ↓ Financial
Processing Layer ↓ Valuation Simulation Engine ↓ News Sentiment Analysis
↓ RAG AI Reasoning Layer ↓ Interactive Dashboard ↓ Automated Investment
Report

------------------------------------------------------------------------

# Core Features

## 1. Stock Data Retrieval

The system retrieves market data using the **yfinance API** including:

-   historical prices
-   company fundamentals
-   valuation metrics
-   sector and industry information

Function: `get_stock_data()`

------------------------------------------------------------------------

## 2. Financial Statement Analysis

Financial statements are parsed and standardized into **Million USD
units**.

Metrics extracted:

-   Total Revenue
-   Gross Profit
-   Operating Income
-   Net Income

Function: `get_financial_statements()`

------------------------------------------------------------------------

## 3. Technical Analysis Dashboard

Technical indicators calculated using **pandas-ta**:

-   MA(5)
-   MA(20)
-   MA(60)
-   MA(90)
-   MA(248)

Visualizations include:

-   candlestick chart
-   moving average overlays
-   trend analysis

------------------------------------------------------------------------

## 4. Valuation Scenario Simulation

A rule-based valuation engine evaluates companies using:

-   P/E ratio
-   revenue growth
-   profit margin
-   market capitalization
-   leverage
-   sector context
-   news sentiment

Three price scenarios are produced:

-   Bullish
-   Neutral
-   Bearish

Functions:

-   `dynamic_valuation_model()`
-   `simulated_valuation()`

------------------------------------------------------------------------

## 5. News Sentiment Intelligence

Recent news headlines are analyzed to determine sentiment.

Sentiment classification:

Positive = +1\
Neutral = 0\
Negative = -1

Aggregated scores influence valuation scenarios.

Functions:

-   `get_recent_news()`
-   `summarize_news_sentiment()`

------------------------------------------------------------------------

# RAG-Based AI Analysis Engine

StockMaster includes a Retrieval-Augmented Generation system.

Pipeline:

User Question\
↓\
Query Expansion\
↓\
Document Retrieval\
↓\
Context Assembly\
↓\
LLM Reasoning\
↓\
Cited Answer

Functions:

-   `build_chat_knowledge_base()`
-   `retrieve_chat_documents()`
-   `generate_chat_answer_with_citations()`

------------------------------------------------------------------------

# AI Investment Plan Generator

Google Gemini is used to generate structured investment analysis using:

-   company fundamentals
-   valuation scenarios
-   financial signals
-   sentiment analysis

Function: `get_ai_investment_plan()`

The output resembles a professional equity research memo.

------------------------------------------------------------------------

# AI Stock Chatbot

Users can ask follow-up questions about a selected ticker.

Example questions:

-   What are the key risks of this company?
-   Explain the growth drivers.
-   How does sentiment affect the valuation scenarios?
-   What is the company's business model?

The chatbot answers using the RAG pipeline.

------------------------------------------------------------------------

# DOCX Report Export

The system can export a professional research report in `.docx` format.

The report includes:

-   company fundamentals
-   financial analysis
-   valuation scenarios
-   AI-generated commentary
-   risk discussion

Function: `generate_docx_report()`

------------------------------------------------------------------------

# Project Structure

StockMaster/

├── stock_master_app_fixed.py\
├── stock_data_utils_fixed.py\
├── rag_chat_pipeline.py\
├── run_dashboard.bat\
└── README.md

------------------------------------------------------------------------

# Technology Stack

Frontend

-   Dash
-   Plotly
-   Bootstrap

Backend

-   Python
-   pandas
-   yfinance
-   pandas-ta
-   requests

AI / NLP

-   Google Gemini
-   Retrieval-Augmented Generation
-   TF-IDF similarity ranking

Report Generation

-   python-docx

------------------------------------------------------------------------

# Workflow

1.  User enters a stock ticker.
2.  Market data and fundamentals are retrieved.
3.  Financial statements are processed.
4.  Technical indicators are calculated.
5.  Valuation scenarios are simulated.
6.  News sentiment is analyzed.
7.  AI generates investment insights.
8.  User can ask follow-up questions.
9.  Report can be exported to DOCX.

------------------------------------------------------------------------

# Installation

pip install dash plotly pandas yfinance pandas-ta requests
google-generativeai python-docx scikit-learn

------------------------------------------------------------------------

# Environment Variables

GEMINI_API_KEY=your_api_key\
NEWS_API_KEY=your_news_api_key

------------------------------------------------------------------------

# Running the Application

python stock_master_app_fixed.py

or

run_dashboard.bat

Open:

http://127.0.0.1:8050/

------------------------------------------------------------------------

# Future Improvements

-   Vector database RAG
-   Hybrid retrieval
-   Portfolio analytics
-   Multi-stock comparison
-   Options analytics
-   Rich DOCX export with charts

------------------------------------------------------------------------

# Disclaimer

This project is for research and educational purposes only and does not
constitute investment advice.
