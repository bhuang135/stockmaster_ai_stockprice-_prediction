# Filename: stock_master_app.py
# Updated: Compatible with RAG-grounded get_ai_investment_plan

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
from dash import dcc, html, dash_table, callback_context

from dash.dash_table.Format import Format, Group
from stock_data_utils_fixed import (
    get_stock_data,
    simulated_valuation,
    get_ai_investment_plan,
    generate_docx_report,
    get_financial_statements
)

def get_dash_column_type(df_column):
    if pd.api.types.is_numeric_dtype(df_column):
        return "numeric"
    elif pd.api.types.is_datetime64_any_dtype(df_column):
        return "datetime"
    else:
        return "text"


METRICS_EXPLANATION = [
    {'Metric': 'P/E Ratio (TTM)', 'How it\'s Calculated': 'Stock Price / Earnings Per Share (EPS) over the past twelve months (TTM).', 'Data Period': 'Past 12 Months', 'Market Significance': 'Value indicator. Measures valuation levels.'},
    {'Metric': 'Forward P/E', 'How it\'s Calculated': 'Stock Price / Projected Earnings Per Share (EPS) for the next twelve months.', 'Data Period': 'Next 12 Months (Earnings Forecast)', 'Market Significance': 'Expectation indicator. Based on analyst consensus forecasts for future earnings.'},
    {'Metric': 'Revenue Growth (YoY)', 'How it\'s Calculated': 'Percentage change between total revenue of the latest reporting period and the same period last year.', 'Data Period': 'Latest Quarter/Year (YoY Comparison)', 'Market Significance': 'Growth indicator. Measures the speed of the company\'s revenue growth.'},
    {'Metric': 'Profit Margin', 'How it\'s Calculated': 'Net Income / Total Revenue.', 'Data Period': 'Past 12 Months (TTM)', 'Market Significance': 'Efficiency indicator. Measures how much net profit a company generates per $1 of revenue.'},
    {'Metric': 'ROE (Return on Equity)', 'How it\'s Calculated': 'Net Income / Shareholder\'s Equity.', 'Data Period': 'Past 12 Months (TTM)', 'Market Significance': 'Efficiency indicator. Measures a company\'s ability to generate profit using shareholders\' capital.'},
    {'Metric': 'Debt/Equity', 'How it\'s Calculated': 'Total Debt / Shareholder\'s Equity.', 'Data Period': 'Latest Quarter/Year (Balance Sheet data)', 'Market Significance': 'Risk indicator. Measures the proportion of debt relative to equity in a company\'s capital structure.'},
    {'Metric': 'Market Cap ($B)', 'How it\'s Calculated': 'Stock Price * Total Shares Outstanding.', 'Data Period': 'Real-time', 'Market Significance': 'Scale indicator. Measures the total market value of the company.'},
    {'Metric': 'Data Period', 'How it\'s Calculated': 'Historical stock price data range: Past 5 years (5Y).', 'Data Period': 'Past 5 Years (Used for Candlestick & Technical Analysis)', 'Market Significance': 'Historical analysis timeframe.'}
]
METRICS_COLUMNS = [{"name": i, "id": i} for i in ['Metric', 'How it\'s Calculated', 'Data Period', 'Market Significance']]

app = dash.Dash(__name__, external_stylesheets=['https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css'])

app.layout = html.Div(
    className='container-fluid',
    style={'padding': '30px', 'backgroundColor': '#f8f9fa'},
    children=[

        html.Div(
            className='row mb-4',
            children=[
                html.Div(
                    className='col-12 text-center',
                    children=[
                        html.H1(
                            "StockMaster Integrated Analysis Dashboard",
                            style={'color': '#1E3A8A', 'fontWeight': 'bold'}
                        ),
                        html.P(
                            "Professional US Stock Data Scraping, AI Investment Advice & Dynamic Price Projection System",
                            className='text-muted'
                        )
                    ]
                )
            ]
        ),

        html.Div(
            className='row justify-content-center mb-5',
            children=[
                html.Div(
                    className='col-md-5 p-4 bg-white shadow-sm rounded',
                    children=[
                        html.Label(
                            "Enter Stock Ticker (e.g., AAPL, NVDA, TSLA):",
                            className='font-weight-bold'
                        ),
                        dcc.Input(
                            id='ticker-input',
                            value='AAPL',
                            type='text',
                            debounce=True,
                            className='form-control form-control-lg'
                        ),
                        html.Small(
                            "Tip: Press Enter or click outside the box to trigger automatic analysis after modification.",
                            className='text-info'
                        )
                    ]
                )
            ]
        ),

        html.Div(
            children=[

                html.Div(
                    id='error-message',
                    className='alert alert-danger',
                    style={'display': 'none'}
                ),

                html.Div(
                    className='mb-5',
                    children=[
                        html.H3(
                            "Core Financial Data (TTM)",
                            style={
                                'color': '#059669',
                                'borderLeft': '5px solid #059669',
                                'paddingLeft': '10px'
                            }
                        ),
                        dcc.Loading(
                            id='loading-fundamental-output',
                            type='default',
                            children=html.Div(
                                id='fundamental-output',
                                className='bg-white shadow-sm rounded p-3'
                            )
                        )
                    ]
                ),

                html.Div(
                    className='row mb-5',
                    children=[
                        html.Div(
                            className='col-md-6',
                            children=[
                                html.H3(
                                    "5-Year Summary P&L (Unit: $M)",
                                    style={
                                        'color': '#1E40AF',
                                        'borderLeft': '5px solid #1E40AF',
                                        'paddingLeft': '10px'
                                    }
                                ),
                                dcc.Loading(
                                    id='loading-pl-table-output',
                                    type='default',
                                    children=html.Div(
                                        id='pl-table-output',
                                        className='bg-white shadow-sm rounded p-3'
                                    )
                                )
                            ]
                        ),
                        html.Div(
                            className='col-md-6',
                            children=[
                                html.H3(
                                    "Financial Growth Trends",
                                    style={
                                        'color': '#1E40AF',
                                        'borderLeft': '5px solid #1E40AF',
                                        'paddingLeft': '10px'
                                    }
                                ),
                                dcc.Loading(
                                    id='loading-pl-trend-chart',
                                    type='default',
                                    children=dcc.Graph(
                                        id='pl-trend-chart',
                                        config={'responsive': True},
                                        style={'height': '400px'}
                                    )
                                )
                            ]
                        )
                    ]
                ),

                html.Div(
                    className='mb-5',
                    style={'minHeight': '600px'},
                    children=[
                        html.H3(
                            "Technical Analysis (MA Multi-Moving Average System)",
                            style={
                                'color': '#06B6D4',
                                'borderLeft': '5px solid #06B6D4',
                                'paddingLeft': '10px'
                            }
                        ),
                        dcc.Loading(
                            id='loading-price-chart',
                            type='default',
                            children=dcc.Graph(
                                id='price-chart',
                                config={'responsive': True},
                                style={'height': '550px'}
                            )
                        )
                    ]
                ),

                html.Div(
                    className='row',
                    children=[

                        html.Div(
                            className='col-md-6 mb-5',
                            children=[
                                html.H3(
                                    "Simulated Price Forward Assessment",
                                    style={
                                        'color': '#DC2626',
                                        'borderLeft': '5px solid #DC2626',
                                        'paddingLeft': '10px'
                                    }
                                ),

                                html.Div(
                                    className='p-4 shadow-sm rounded bg-white h-100',
                                    children=[
                                        dcc.Loading(
                                            id='loading-prediction-output',
                                            type='default',
                                            children=html.Div(
                                                id='prediction-output',
                                                className='mb-4'
                                            )
                                        ),

                                        html.Hr(),

                                        html.H5(
                                            "AI Investment Chatbot",
                                            style={
                                                'color': '#1F2937',
                                                'borderLeft': '4px solid #1F2937',
                                                'paddingLeft': '10px',
                                                'marginTop': '20px',
                                                'marginBottom': '15px'
                                            }
                                        ),

                                        dcc.Textarea(
                                            id='chat-input',
                                            placeholder='Ask anything about this stock...',
                                            style={
                                                'width': '100%',
                                                'height': '120px',
                                                'marginBottom': '12px'
                                            }
                                        ),

                                        html.Button(
                                            "Ask AI",
                                            id="chat-submit",
                                            className="btn btn-primary"
                                        ),

                                        dcc.Loading(
                                            id='loading-chat-output',
                                            type='default',
                                            children=html.Div(
                                                id="chat-output",
                                                className="p-3 mt-3 bg-light rounded",
                                                style={'minHeight': '120px'}
                                            )
                                        )
                                    ]
                                )
                            ]
                        ),

                        html.Div(
                            className='col-md-6 mb-5',
                            children=[
                                html.H3(
                                    "Gemini AI Institutional Investment Plan (RAG-grounded + Citations)",
                                    style={
                                        'color': '#8B5CF6',
                                        'borderLeft': '5px solid #8B5CF6',
                                        'paddingLeft': '10px'
                                    }
                                ),

                                dcc.Loading(
                                    id='loading-ai-investment-plan',
                                    type='default',
                                    children=html.Div(
                                        id='ai-investment-plan',
                                        className='p-4 shadow-sm rounded bg-white h-100'
                                    )
                                ),

                                html.Div(
                                    className='mt-3 text-right',
                                    children=[
                                        html.Button(
                                            "Download Professional Investment Report (.docx)",
                                            id="btn-download-docx",
                                            className="btn btn-outline-primary"
                                        ),
                                        dcc.Download(id="download-report-docx")
                                    ]
                                )
                            ]
                        )
                    ]
                )
            ]
        ),

        html.H3(
            "Core Financial Metrics Detail",
            style={
                'color': '#1E3A8A',
                'borderLeft': '5px solid #1E3A8A',
                'paddingLeft': '10px'
            }
        ),

        dash_table.DataTable(
            id='metrics-explanation-table',
            columns=METRICS_COLUMNS,
            data=METRICS_EXPLANATION,
            style_header={'backgroundColor': '#EBF4FF', 'fontWeight': 'bold'},
            style_cell={'textAlign': 'left', 'padding': '12px', 'fontSize': '13px'},
            style_data={'whiteSpace': 'normal', 'height': 'auto'},
            page_size=11
        )
    ]
)


@app.callback(
    [Output('fundamental-output', 'children'),
     Output('price-chart', 'figure'),
     Output('prediction-output', 'children'),
     Output('ai-investment-plan', 'children'),
     Output('pl-table-output', 'children'),
     Output('pl-trend-chart', 'figure'),
     Output('error-message', 'children'),
     Output('error-message', 'style')],
    [Input('ticker-input', 'value')]
)
def update_dashboard(ticker):
    # Output count must always match 8 outputs
    if not ticker:
        return (dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update,
                "Please enter a valid stock ticker.", {'display': 'block'})

    ticker = ticker.upper()

    try:
        # 1) Fetch data
        df, fundamentals = get_stock_data(ticker)
        if df.empty:
            return (dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                    dash.no_update, dash.no_update,
                    f"Unable to fetch historical data for {ticker}.", {'display': 'block'})

        # 2) Fundamentals table
        fund_data = [
            {'Metric': k, 'Value': v}
            for k, v in fundamentals.items()
            if k not in ['Data Source', 'Data Period', 'Symbol', 'Company Name', 'Business Summary', 'Website']
        ]
        fund_table = dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in ['Metric', 'Value']],
            data=fund_data,
            style_cell={'textAlign': 'left', 'padding': '8px'}
        )

        # 3) Candlestick chart
        fig = go.Figure(data=[
            go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name='Price'
            )
        ])
        colors = {'MA(5)': 'orange', 'MA(20)': 'yellow', 'MA(60)': 'green', 'MA(90)': 'red', 'MA(248)': 'purple'}
        for ma_val, color in colors.items():
            window = int(ma_val.split('(')[1].split(')')[0])
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Close'].rolling(window=window).mean(),
                mode='lines',
                name=ma_val,
                line={'color': color, 'width': 1}
            ))
        fig.update_layout(template="plotly_white", height=550, xaxis_rangeslider_visible=False, title=f"{ticker} 5-Year Trend Analysis")

        # 4) Simulation + RAG-grounded AI plan
        current_price = float(df['Close'].iloc[-1])
        current_close, positive_price, neutral_price, negative_price, model_desc, valuation_debug = simulated_valuation(df, fundamentals)
        ai_text = get_ai_investment_plan(
            ticker,
            fundamentals,
            current_close,
            model_desc,
            positive_price,
            neutral_price,
            negative_price
        )

        pred_md = dcc.Markdown(
            f"**Model Basis:** {model_desc}\n\n"
            f"**Current Close:** `${current_close:.2f}`\n\n"
            f"**Positive Scenario:** `${positive_price:.2f}`\n\n"
            f"**Neutral Scenario:** `${neutral_price:.2f}`\n\n"
            f"**Negative Scenario:** `${negative_price:.2f}`",
            dangerously_allow_html=True
        )
        ai_md = dcc.Markdown(ai_text)

        # 5) P&L
        df_pl = get_financial_statements(ticker)

        if not df_pl.empty:
            pl_table = dash_table.DataTable(
                columns=[
                    {"name": str(i), "id": str(i), "type": "numeric" if i != "Year" else "any",
                     "format": Format(group=Group.yes) if i != "Year" else None}
                    for i in df_pl.columns
                ],
                data=df_pl.to_dict('records'),
                style_cell={'textAlign': 'center', 'padding': '10px'},
                style_header={'backgroundColor': '#F3F4F6', 'fontWeight': 'bold'},
                style_data_conditional=[{
                    'if': {'column_type': 'numeric'},
                    'textAlign': 'right'
                }]
            )

            pl_fig = go.Figure()
            pl_fig.add_trace(go.Bar(x=df_pl['Year'], y=df_pl['Total Revenue'], name='Total Revenue'))
            pl_fig.add_trace(go.Scatter(x=df_pl['Year'], y=df_pl['Net Income'], name='Net Income', mode='lines+markers'))
            pl_fig.update_layout(template="plotly_white", barmode='group', height=400, margin=dict(t=20, b=20, l=20, r=20))
        else:
            pl_table = html.Div("No financial statement data available.")
            pl_fig = go.Figure()

        return fund_table, fig, pred_md, ai_md, pl_table, pl_fig, "", {'display': 'none'}

    except Exception as e:
        return (dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update,
                f"System Error: {str(e)}", {'display': 'block'})


@app.callback(
    Output("download-report-docx", "data"),
    Input("btn-download-docx", "n_clicks"),
    [State('ticker-input', 'value'),
     State('ai-investment-plan', 'children')],
    prevent_initial_call=True,
)
def handle_docx_download(n_clicks, ticker, ai_md_content):
    if not n_clicks or not ticker:
        return dash.no_update

    try:
        _, fundamentals = get_stock_data(ticker)

        ai_text = ""
        if ai_md_content and isinstance(ai_md_content, dict) and 'props' in ai_md_content:
            ai_text = ai_md_content['props'].get('children', "")

        file_stream = generate_docx_report(ticker.upper(), fundamentals, ai_text)
        return dcc.send_bytes(file_stream.getvalue(), f"StockMaster_{ticker.upper()}_Report.docx")
    except Exception as e:
        print(f"Download Error: {e}")
        return dash.no_update

@app.callback(
    [Output("chat-output", "children"),
     Output("chat-input", "value")],
    [Input("chat-submit", "n_clicks"),
     Input("ticker-input", "value")],
    [State("chat-input", "value")],
    prevent_initial_call=True
)
def chat_with_ai(n_clicks, ticker, question):
    ctx = callback_context

    if not ctx.triggered:
        return dash.no_update, dash.no_update

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # 1)  chat input + chat result
    if trigger_id == "ticker-input":
        return "", ""

    # 2) Ask AI
    if trigger_id == "chat-submit":
        if not question:
            return "Please enter a question.", dash.no_update

        try:
            from stock_data_utils_fixed import get_ai_chat_response
            response = get_ai_chat_response(ticker, question)
            return dcc.Markdown(response), dash.no_update
        except Exception as e:
            return f"Error: {str(e)}", dash.no_update

    return dash.no_update, dash.no_update

if __name__ == '__main__':
    app.run(debug=True)