
# ================================================================
# INSTITUTIONAL DISTRESS INTELLIGENCE ENGINE — VALIDATED EDITION
# ================================================================
# UI inspired by the attached editorial / scrapbook-style reference:
# dusty blue background, cream paper panels, striped header/footer,
# serif display typography, numbered workflow cards, clean alignment.
#
# DATA ARCHITECTURE
# 1) Screener.in = primary convenience source for ratios / working-capital days
# 2) Yahoo Finance = independent secondary source for raw annual statements
# 3) Cross-check comparable values and flag discrepancies
# 4) Never replace missing data with invented fallback values
# 5) Only validated / available inputs feed the distress models
# 6) Gemini interprets the validated analytics; it does NOT invent data
#
# NOTE:
# - This is a classroom / analytical screening engine, not a credit rating.
# - Data-quality score measures validation coverage, not probability of truth.
# - Official annual report / NSE / BSE filings remain the final authority.
# ================================================================

import subprocess, sys

# Pin the UI framework so a future rerun cannot silently pull a new
# Gradio version and change the app styling/DOM. Gradio 6.29.1 is the
# current stable release used for this build.
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gradio==6.29.1", "google-genai", "yfinance", "beautifulsoup4", "requests", "pandas", "numpy"])

import gradio as gr
import pandas as pd
import numpy as np
import requests
import re
import math
import json
import time
import random
import html
import traceback
from bs4 import BeautifulSoup
import yfinance as yf
from google import genai
from google.colab import userdata


# ================================================================
# 1. GEMINI CONFIG
# ================================================================

API_KEY = ""

try:
    API_KEY = userdata.get("GEMINI_API_KEY")
except Exception as e:
    print(f"Could not read GEMINI_API_KEY from Colab Secrets: {e}")

# Current text-generation models only.
# Do NOT add TTS / audio / live / image model IDs here.
GEMINI_TEXT_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

ACTIVE_GEMINI_MODEL = None

# Reset the cached model on every clean script execution so a previous
# notebook run cannot leave stale Gemini state behind.


def get_gemini_client():
    global API_KEY

    if not API_KEY:
        try:
            API_KEY = userdata.get("GEMINI_API_KEY")
        except Exception:
            pass

    if not API_KEY:
        raise ValueError(
            "GEMINI_API_KEY not found. Add it to Colab Secrets "
            "and enable Notebook access."
        )

    return genai.Client(api_key=API_KEY)


def classify_gemini_error(error):
    t = str(error).upper()

    for code, keys in {
        "503": ["503", "UNAVAILABLE"],
        "429": ["429", "RESOURCE_EXHAUSTED"],
        "404": ["404", "NOT_FOUND"],
        "401": ["401", "UNAUTHENTICATED"],
        "403": ["403", "PERMISSION_DENIED"],
        "500": ["500", "INTERNAL"],
        "502": ["502", "BAD_GATEWAY"],
        "504": ["504", "DEADLINE_EXCEEDED"],
    }.items():
        if any(k in t for k in keys):
            return code

    return "OTHER"


def generate_with_retry(client, model_name, prompt, max_attempts=3):
    last_error = None

    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            text_response = getattr(response, "text", None)

            if text_response:
                return text_response

            raise ValueError("Gemini returned an empty response.")

        except Exception as e:
            last_error = e
            code = classify_gemini_error(e)

            # Client errors: do not waste retries.
            if code in ["400", "401", "403", "404"]:
                raise

            # Transient errors: exponential backoff.
            if code in ["429", "500", "502", "503", "504"]:
                if attempt < max_attempts - 1:
                    delay = min(20, (2 ** (attempt + 1))) + random.uniform(0.25, 1.0)
                    time.sleep(delay)
                    continue

            raise

    raise last_error


def call_gemini_api(prompt):
    global ACTIVE_GEMINI_MODEL

    try:
        client = get_gemini_client()
    except Exception as e:
        return f"⚠️ **Gemini configuration error**\n\n{str(e)}"

    models_to_try = []

    if ACTIVE_GEMINI_MODEL:
        models_to_try.append(ACTIVE_GEMINI_MODEL)

    for model in GEMINI_TEXT_MODELS:
        if model not in models_to_try:
            models_to_try.append(model)

    errors = []

    for model_name in models_to_try:
        try:
            result = generate_with_retry(
                client,
                model_name,
                prompt,
                max_attempts=3
            )

            if result:
                ACTIVE_GEMINI_MODEL = model_name
                return result

        except Exception as e:
            errors.append(
                f"{model_name}: {classify_gemini_error(e)} — {str(e)}"
            )

    return (
        "⚠️ **Gemini is temporarily unavailable.**\n\n"
        "The validated quantitative analysis is still available. "
        "The AI commentary layer could not be generated.\n\n"
        + "\n".join(f"• {x}" for x in errors)
    )


# ================================================================
# 2. NIFTY 50 / COMMON INDIAN TICKERS
# ================================================================

NIFTY_50_MAP = {
    "Adani Enterprises Ltd (ADANIENT)": "ADANIENT",
    "Adani Ports & SEZ (ADANIPORTS)": "ADANIPORTS",
    "Apollo Hospitals (APOLLOHOSP)": "APOLLOHOSP",
    "Asian Paints (ASIANPAINT)": "ASIANPAINT",
    "Axis Bank (AXISBANK)": "AXISBANK",
    "Bajaj Auto (BAJAJ-AUTO)": "BAJAJ-AUTO",
    "Bajaj Finance (BAJFINANCE)": "BAJFINANCE",
    "Bajaj Finserv (BAJAJFINSV)": "BAJAJFINSV",
    "Bharat Electronics (BEL)": "BEL",
    "Bharti Airtel (BHARTIARTL)": "BHARTIARTL",
    "Cipla (CIPLA)": "CIPLA",
    "Coal India (COALINDIA)": "COALINDIA",
    "Dr. Reddy's Laboratories (DRREDDY)": "DRREDDY",
    "Eicher Motors (EICHERMOT)": "EICHERMOT",
    "Grasim Industries (GRASIM)": "GRASIM",
    "HCL Technologies (HCLTECH)": "HCLTECH",
    "HDFC Bank (HDFCBANK)": "HDFCBANK",
    "HDFC Life Insurance (HDFCLIFE)": "HDFCLIFE",
    "Hindalco Industries (HINDALCO)": "HINDALCO",
    "Hindustan Unilever (HINDUNILVR)": "HINDUNILVR",
    "ICICI Bank (ICICIBANK)": "ICICIBANK",
    "IndiGo / InterGlobe (INDIGO)": "INDIGO",
    "Infosys (INFY)": "INFY",
    "ITC Ltd (ITC)": "ITC",
    "JSW Steel (JSWSTEEL)": "JSWSTEEL",
    "Kotak Mahindra Bank (KOTAKBANK)": "KOTAKBANK",
    "Larsen & Toubro (LT)": "LT",
    "Mahindra & Mahindra (M&M)": "M&M",
    "Maruti Suzuki (MARUTI)": "MARUTI",
    "NTPC Ltd (NTPC)": "NTPC",
    "Nestle India (NESTLEIND)": "NESTLEIND",
    "ONGC (ONGC)": "ONGC",
    "Power Grid Corp (POWERGRID)": "POWERGRID",
    "Reliance Industries (RELIANCE)": "RELIANCE",
    "SBI Life Insurance (SBILIFE)": "SBILIFE",
    "State Bank of India (SBIN)": "SBIN",
    "Shriram Finance (SHRIRAMFIN)": "SHRIRAMFIN",
    "Sun Pharma (SUNPHARMA)": "SUNPHARMA",
    "Tata Consumer Products (TATACONSUM)": "TATACONSUM",
    "Tata Consultancy Services (TCS)": "TCS",
    "Tata Motors (TMPV)": "TMPV",
    "Tata Steel (TATASTEEL)": "TATASTEEL",
    "Tech Mahindra (TECHM)": "TECHM",
    "Titan Company (TITAN)": "TITAN",
    "Trent Ltd (TRENT)": "TRENT",
    "UltraTech Cement (ULTRACEMCO)": "ULTRACEMCO",
    "Wipro (WIPRO)": "WIPRO",
    "Custom / Other Indian Ticker": "CUSTOM",
}


# ================================================================
# 3. GENERIC HELPERS
# ================================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0 Safari/537.36"
    )
}


def clean_number(value):
    if value is None:
        return None

    s = str(value).strip()

    if not s or s in {"-", "—", "NA", "N/A", "None"}:
        return None

    # Parentheses represent negative numbers in many statements.
    negative = s.startswith("(") and s.endswith(")")

    s = s.replace(",", "")
    s = re.sub(r"[₹$€£]", "", s)
    s = re.sub(r"[^\d.\-]", "", s)

    if not s:
        return None

    try:
        n = float(s)
        return -abs(n) if negative else n
    except Exception:
        return None


def safe_float(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except Exception:
        return None


def pct_difference(a, b):
    a = safe_float(a)
    b = safe_float(b)

    if a is None or b is None:
        return None

    denominator = max(abs(b), 1e-12)
    return abs(a - b) / denominator * 100.0


def first_value(d, aliases):
    for alias in aliases:
        if alias in d and d[alias] is not None:
            return d[alias]
    return None


def find_series_row(df, aliases):
    if df is None or df.empty:
        return None

    normalized = {
        re.sub(r"[^a-z0-9]", "", str(idx).lower()): idx
        for idx in df.index
    }

    for alias in aliases:
        key = re.sub(r"[^a-z0-9]", "", alias.lower())

        if key in normalized:
            return normalized[key]

    # More forgiving contains-match.
    for alias in aliases:
        key = re.sub(r"[^a-z0-9]", "", alias.lower())

        for norm_key, original in normalized.items():
            if key and (key in norm_key or norm_key in key):
                return original

    return None


def get_series_values(df, aliases, n=2):
    row = find_series_row(df, aliases)

    if row is None:
        return [None] * n

    values = []

    for col in list(df.columns)[:n]:
        try:
            values.append(safe_float(df.loc[row, col]))
        except Exception:
            values.append(None)

    while len(values) < n:
        values.append(None)

    return values[:n]


def get_year_from_label(label):
    s = str(label)

    # Match 4 digit years.
    years = re.findall(r"(20\d{2})", s)
    if years:
        return int(years[-1])

    # Match Mar 25 / Mar-25.
    m = re.search(r"[A-Za-z]{3,9}\s*[-/]?\s*(\d{2})$", s)
    if m:
        yy = int(m.group(1))
        return 2000 + yy

    return None


# ================================================================
# 4. SCREENER EXTRACTION
# ================================================================

def parse_screener_table(section):
    """
    Returns:
        {
            "row name": {
                2026: value,
                2025: value,
                ...
            }
        }
    """

    data = {}

    if section is None:
        return data

    table = section.find("table", class_="data-table")
    if table is None:
        return data

    rows = table.find_all("tr")

    # Header years.
    header_cells = rows[0].find_all(["th", "td"]) if rows else []
    year_cols = {}

    for i, cell in enumerate(header_cells):
        year = get_year_from_label(cell.get_text(" ", strip=True))
        if year:
            year_cols[i] = year

    for row in rows[1:]:
        cells = row.find_all(["th", "td"])

        if len(cells) < 2:
            continue

        row_name = cells[0].get_text(" ", strip=True).lower()

        if not row_name:
            continue

        series = {}

        for idx, year in year_cols.items():
            if idx < len(cells):
                value = clean_number(
                    cells[idx].get_text(" ", strip=True)
                )
                if value is not None:
                    series[year] = value

        if series:
            data[row_name] = series

    return data


def latest_and_previous(series):
    if not series:
        return None, None, None, None

    years = sorted(series.keys(), reverse=True)

    latest_year = years[0]
    latest_value = series[latest_year]

    previous_year = years[1] if len(years) > 1 else None
    previous_value = (
        series[previous_year]
        if previous_year is not None
        else None
    )

    return (
        latest_value,
        previous_value,
        latest_year,
        previous_year
    )


def scrape_screener(symbol):
    url = (
        f"https://www.screener.in/company/"
        f"{symbol.upper()}/consolidated/"
    )

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=25
    )

    if response.status_code != 200:
        url = f"https://www.screener.in/company/{symbol.upper()}/"
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=25
        )

    if response.status_code != 200:
        raise ValueError(
            f"Screener.in returned HTTP {response.status_code}"
        )

    soup = BeautifulSoup(response.text, "html.parser")

    top_ratios = {}

    for item in soup.find_all(
        "li",
        class_="flex flex-space-between"
    ):
        name_el = item.find("span", class_="name")
        val_el = item.find("span", class_="number")

        if name_el and val_el:
            name = name_el.get_text(" ", strip=True).lower()
            value = clean_number(
                val_el.get_text(" ", strip=True)
            )
            if value is not None:
                top_ratios[name] = value

    sections = {
        "profit-loss": parse_screener_table(
            soup.find("section", id="profit-loss")
        ),
        "balance-sheet": parse_screener_table(
            soup.find("section", id="balance-sheet")
        ),
        "cash-flow": parse_screener_table(
            soup.find("section", id="cash-flow")
        ),
        "ratios": parse_screener_table(
            soup.find("section", id="ratios")
        ),
    }

    def get_latest(section_name, aliases):
        for alias in aliases:
            normalized = alias.lower()
            for row_name, series in sections[section_name].items():
                if normalized == row_name or normalized in row_name:
                    return latest_and_previous(series)
        return None, None, None, None

    return {
        "symbol": symbol,
        "url": url,
        "top_ratios": top_ratios,
        "sections": sections,
        "get_latest": get_latest,
    }


# ================================================================
# 5. YAHOO FINANCE EXTRACTION
# ================================================================

def fetch_yahoo(symbol):
    ticker_symbol = symbol.upper().strip()

    if not ticker_symbol.endswith((".NS", ".BO")):
        ticker_symbol += ".NS"

    stock = yf.Ticker(ticker_symbol)

    # These calls may be slow, but they give independent raw statement data.
    info = {}
    try:
        info = stock.info or {}
    except Exception:
        info = {}

    try:
        financials = stock.financials
    except Exception:
        financials = pd.DataFrame()

    try:
        balance_sheet = stock.balance_sheet
    except Exception:
        balance_sheet = pd.DataFrame()

    try:
        cashflow = stock.cashflow
    except Exception:
        cashflow = pd.DataFrame()

    # Ensure newest columns first.
    for df_name in ["financials", "balance_sheet", "cashflow"]:
        pass

    if not financials.empty:
        financials = financials.reindex(
            sorted(financials.columns, reverse=True),
            axis=1
        )

    if not balance_sheet.empty:
        balance_sheet = balance_sheet.reindex(
            sorted(balance_sheet.columns, reverse=True),
            axis=1
        )

    if not cashflow.empty:
        cashflow = cashflow.reindex(
            sorted(cashflow.columns, reverse=True),
            axis=1
        )

    sector = info.get("sector") or ""
    industry = info.get("industry") or ""
    company_name = (
        info.get("longName")
        or info.get("shortName")
        or symbol
    )

    market_cap = safe_float(info.get("marketCap"))

    return {
        "ticker": ticker_symbol,
        "company_name": company_name,
        "sector": sector,
        "industry": industry,
        "market_cap_inr": market_cap,
        "financials": financials,
        "balance_sheet": balance_sheet,
        "cashflow": cashflow,
        "url": f"https://finance.yahoo.com/quote/{ticker_symbol}/",
    }


# ================================================================
# 6. RAW YAHOO METRIC EXTRACTION
# ================================================================

FINANCIAL_ALIASES = {
    "revenue": [
        "Total Revenue",
        "Operating Revenue",
        "Revenue"
    ],
    "gross_profit": [
        "Gross Profit"
    ],
    "ebit": [
        "EBIT"
    ],
    "ebitda": [
        "EBITDA"
    ],
    "interest_expense": [
        "Interest Expense",
        "Interest Expense Non Operating",
        "Net Non Operating Interest Income Expense"
    ],
    "net_income": [
        "Net Income",
        "Net Income Common Stockholders"
    ],
    "sga": [
        "Selling General And Administration",
        "Selling General Administrative",
        "Selling General and Administrative"
    ],
    "cost_of_revenue": [
        "Cost Of Revenue",
        "Cost of Revenue",
        "Total Expenses"
    ],
    "depreciation": [
        "Depreciation",
        "Depreciation And Amortization"
    ],
}

BALANCE_ALIASES = {
    "current_assets": [
        "Current Assets"
    ],
    "current_liabilities": [
        "Current Liabilities"
    ],
    "cash": [
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
        "Cash Financial"
    ],
    "accounts_receivable": [
        "Accounts Receivable",
        "Receivables"
    ],
    "inventory": [
        "Inventory"
    ],
    "accounts_payable": [
        "Accounts Payable",
        "Payables"
    ],
    "total_assets": [
        "Total Assets"
    ],
    "total_liabilities": [
        "Total Liabilities Net Minority Interest",
        "Total Liabilities"
    ],
    "stockholders_equity": [
        "Stockholders Equity",
        "Common Stock Equity",
        "Total Equity Gross Minority Interest"
    ],
    "retained_earnings": [
        "Retained Earnings",
        "Retained Earnings Accumulated Deficit"
    ],
    "total_debt": [
        "Total Debt"
    ],
    "long_term_debt": [
        "Long Term Debt",
        "Long Term Debt And Capital Lease Obligation"
    ],
    "net_ppe": [
        "Net PPE",
        "Net Property Plant And Equipment"
    ],
    "working_capital": [
        "Working Capital"
    ],
}

CASHFLOW_ALIASES = {
    "operating_cash_flow": [
        "Operating Cash Flow",
        "Total Cash From Operating Activities"
    ],
}


def get_df_metric(df, aliases, index=0):
    values = get_series_values(df, aliases, n=3)
    return values[index] if index < len(values) else None


# ================================================================
# 7. BUILD VALIDATED METRICS
# ================================================================

def build_company_dataset(symbol):
    screener_error = None
    yahoo_error = None

    screener = None
    yahoo = None

    try:
        screener = scrape_screener(symbol)
    except Exception as e:
        screener_error = str(e)

    try:
        yahoo = fetch_yahoo(symbol)
    except Exception as e:
        yahoo_error = str(e)

    if screener is None and yahoo is None:
        raise RuntimeError(
            "Both Screener.in and Yahoo Finance failed.\n\n"
            f"Screener: {screener_error}\n"
            f"Yahoo Finance: {yahoo_error}"
        )

    # ------------------------------------------------------------
    # Company metadata
    # ------------------------------------------------------------

    company_name = (
        yahoo.get("company_name")
        if yahoo else symbol
    ) or symbol

    sector = (
        yahoo.get("sector")
        if yahoo else ""
    ) or ""

    industry = (
        yahoo.get("industry")
        if yahoo else ""
    ) or ""

    is_financial = (
        "financial" in sector.lower()
        or "bank" in industry.lower()
        or "insurance" in industry.lower()
    )

    # ------------------------------------------------------------
    # Yahoo raw statement values
    # ------------------------------------------------------------

    yf_fin = yahoo["financials"] if yahoo else pd.DataFrame()
    yf_bs = yahoo["balance_sheet"] if yahoo else pd.DataFrame()
    yf_cf = yahoo["cashflow"] if yahoo else pd.DataFrame()

    # Latest / prior year raw values.
    yf_revenue = get_series_values(
        yf_fin,
        FINANCIAL_ALIASES["revenue"],
        2
    )

    yf_gross_profit = get_series_values(
        yf_fin,
        FINANCIAL_ALIASES["gross_profit"],
        2
    )

    yf_ebit = get_series_values(
        yf_fin,
        FINANCIAL_ALIASES["ebit"],
        2
    )

    yf_interest = get_series_values(
        yf_fin,
        FINANCIAL_ALIASES["interest_expense"],
        2
    )

    yf_net_income = get_series_values(
        yf_fin,
        FINANCIAL_ALIASES["net_income"],
        2
    )

    yf_sga = get_series_values(
        yf_fin,
        FINANCIAL_ALIASES["sga"],
        2
    )

    yf_cogs = get_series_values(
        yf_fin,
        FINANCIAL_ALIASES["cost_of_revenue"],
        2
    )

    yf_depreciation = get_series_values(
        yf_fin,
        FINANCIAL_ALIASES["depreciation"],
        2
    )

    yf_current_assets = get_series_values(
        yf_bs,
        BALANCE_ALIASES["current_assets"],
        2
    )

    yf_current_liabilities = get_series_values(
        yf_bs,
        BALANCE_ALIASES["current_liabilities"],
        2
    )

    yf_cash = get_series_values(
        yf_bs,
        BALANCE_ALIASES["cash"],
        2
    )

    yf_ar = get_series_values(
        yf_bs,
        BALANCE_ALIASES["accounts_receivable"],
        2
    )

    yf_inventory = get_series_values(
        yf_bs,
        BALANCE_ALIASES["inventory"],
        2
    )

    yf_ap = get_series_values(
        yf_bs,
        BALANCE_ALIASES["accounts_payable"],
        2
    )

    yf_assets = get_series_values(
        yf_bs,
        BALANCE_ALIASES["total_assets"],
        2
    )

    yf_liabilities = get_series_values(
        yf_bs,
        BALANCE_ALIASES["total_liabilities"],
        2
    )

    yf_equity = get_series_values(
        yf_bs,
        BALANCE_ALIASES["stockholders_equity"],
        2
    )

    yf_retained_earnings = get_series_values(
        yf_bs,
        BALANCE_ALIASES["retained_earnings"],
        2
    )

    yf_debt = get_series_values(
        yf_bs,
        BALANCE_ALIASES["total_debt"],
        2
    )

    yf_long_debt = get_series_values(
        yf_bs,
        BALANCE_ALIASES["long_term_debt"],
        2
    )

    yf_ppe = get_series_values(
        yf_bs,
        BALANCE_ALIASES["net_ppe"],
        2
    )

    yf_cfo = get_series_values(
        yf_cf,
        CASHFLOW_ALIASES["operating_cash_flow"],
        2
    )

    # ------------------------------------------------------------
    # Derive missing raw values where accounting identity supports it
    # ------------------------------------------------------------

    for i in range(2):
        if (
            yf_gross_profit[i] is None
            and yf_revenue[i] is not None
            and yf_cogs[i] is not None
        ):
            yf_gross_profit[i] = yf_revenue[i] - abs(yf_cogs[i])

    # ------------------------------------------------------------
    # Screener metrics
    # ------------------------------------------------------------

    screener_top = screener["top_ratios"] if screener else {}

    # Screener statement tables are displayed in INR crore.
    # These are used only as an independent cross-check against
    # the raw annual values from Yahoo Finance.
    screener_revenue = None
    screener_pat = None
    screener_ocf = None
    screener_assets = None
    screener_liabilities = None

    if screener:
        pl = screener["sections"].get("profit-loss", {})
        bs = screener["sections"].get("balance-sheet", {})
        cf = screener["sections"].get("cash-flow", {})

        def screener_latest(sections_dict, candidates):
            for candidate in candidates:
                candidate_norm = re.sub(
                    r"[^a-z0-9]", "", candidate.lower()
                )
                for row_name, series in sections_dict.items():
                    row_norm = re.sub(
                        r"[^a-z0-9]", "", row_name.lower()
                    )
                    if (
                        row_norm == candidate_norm
                        or candidate_norm in row_norm
                    ):
                        return latest_and_previous(series)[0]
            return None

        screener_revenue_cr = screener_latest(
            pl,
            ["sales", "revenue"]
        )
        screener_pat_cr = screener_latest(
            pl,
            ["net profit", "profit after tax", "profit"]
        )
        screener_ocf_cr = screener_latest(
            cf,
            [
                "cash from operating activity",
                "operating activity",
            ]
        )
        screener_assets_cr = screener_latest(
            bs,
            ["total assets"]
        )
        screener_liabilities_cr = screener_latest(
            bs,
            ["total liabilities"]
        )

        if screener_revenue_cr is not None:
            screener_revenue = screener_revenue_cr * 10_000_000

        if screener_pat_cr is not None:
            screener_pat = screener_pat_cr * 10_000_000

        if screener_ocf_cr is not None:
            screener_ocf = screener_ocf_cr * 10_000_000

        if screener_assets_cr is not None:
            screener_assets = screener_assets_cr * 10_000_000

        if screener_liabilities_cr is not None:
            screener_liabilities = screener_liabilities_cr * 10_000_000

    def screener_top_value(*names):
        for name in names:
            if name.lower() in screener_top:
                return screener_top[name.lower()]
        return None

    screener_current_ratio = screener_top_value(
        "current ratio"
    )

    screener_de = screener_top_value(
        "debt to equity",
        "debt/equity"
    )

    screener_roce = screener_top_value(
        "roce"
    )

    screener_pledge = screener_top_value(
        "pledged percentage"
    )

    # Working-capital days.
    screener_dso = None
    screener_dio = None
    screener_dpo = None

    if screener:
        ratios = screener["sections"].get("ratios", {})

        for key, target in [
            ("debtor days", "dso"),
            ("inventory days", "dio"),
            ("days payable", "dpo"),
            ("creditor days", "dpo"),
        ]:
            if key in ratios:
                v, pv, y, py = latest_and_previous(ratios[key])
                if target == "dso" and screener_dso is None:
                    screener_dso = v
                if target == "dio" and screener_dio is None:
                    screener_dio = v
                if target == "dpo" and screener_dpo is None:
                    screener_dpo = v

    # ------------------------------------------------------------
    # Independent Yahoo-derived ratios
    # ------------------------------------------------------------

    yahoo_current_ratio = None
    if yf_current_assets[0] is not None and yf_current_liabilities[0] not in (None, 0):
        yahoo_current_ratio = yf_current_assets[0] / yf_current_liabilities[0]

    yahoo_de = None
    if yf_debt[0] is not None and yf_equity[0] not in (None, 0):
        yahoo_de = yf_debt[0] / yf_equity[0]

    yahoo_interest_coverage = None
    if yf_ebit[0] is not None and yf_interest[0] not in (None, 0):
        yahoo_interest_coverage = yf_ebit[0] / abs(yf_interest[0])

    yahoo_roce = None
    if (
        yf_ebit[0] is not None
        and yf_assets[0] is not None
        and yf_current_liabilities[0] is not None
    ):
        capital_employed = yf_assets[0] - yf_current_liabilities[0]
        if capital_employed > 0:
            yahoo_roce = yf_ebit[0] / capital_employed * 100

    # ------------------------------------------------------------
    # Working-capital days independently derived from raw statements.
    # Use average balances when prior year is available.
    # ------------------------------------------------------------

    def avg_or_single(latest, previous):
        if latest is None:
            return None
        if previous is not None:
            return (latest + previous) / 2
        return latest

    avg_ar = avg_or_single(yf_ar[0], yf_ar[1])
    avg_inventory = avg_or_single(
        yf_inventory[0], yf_inventory[1]
    )
    avg_ap = avg_or_single(yf_ap[0], yf_ap[1])

    dso_derived = None
    dio_derived = None
    dpo_derived = None

    if avg_ar is not None and yf_revenue[0] not in (None, 0):
        dso_derived = avg_ar / abs(yf_revenue[0]) * 365

    if (
        avg_inventory is not None
        and yf_cogs[0] not in (None, 0)
    ):
        dio_derived = avg_inventory / abs(yf_cogs[0]) * 365

    if avg_ap is not None and yf_cogs[0] not in (None, 0):
        dpo_derived = avg_ap / abs(yf_cogs[0]) * 365

    # ------------------------------------------------------------
    # Period label
    # ------------------------------------------------------------

    def df_period(df):
        if df is None or df.empty:
            return None
        try:
            col = df.columns[0]
            return pd.Timestamp(col).strftime("%Y")
        except Exception:
            return None

    period = (
        f"FY{df_period(yf_fin)}"
        if df_period(yf_fin)
        else "Latest available annual data"
    )

    # ------------------------------------------------------------
    # Build core metric dictionary
    # ------------------------------------------------------------

    metrics = {

        "Revenue": {
            "value": yf_revenue[0],
            "unit": "INR",
            "source": "Yahoo Finance annual statement",
            "secondary": screener_revenue,
            "secondary_source": "Screener (₹ crore → INR)",
            "period": period,
        },

        "EBIT": {
            "value": yf_ebit[0],
            "unit": "INR",
            "source": "Yahoo Finance annual statement",
            "secondary": None,
            "secondary_source": "",
            "period": period,
        },

        "PAT / Net Income": {
            "value": yf_net_income[0],
            "unit": "INR",
            "source": "Yahoo Finance annual statement",
            "secondary": screener_pat,
            "secondary_source": "Screener (₹ crore → INR)",
            "period": period,
        },

        "Operating Cash Flow": {
            "value": yf_cfo[0],
            "unit": "INR",
            "source": "Yahoo Finance annual cash flow",
            "secondary": screener_ocf,
            "secondary_source": "Screener (₹ crore → INR)",
            "period": period,
        },

        "Total Assets": {
            "value": yf_assets[0],
            "unit": "INR",
            "source": "Yahoo Finance annual balance sheet",
            "secondary": screener_assets,
            "secondary_source": "Screener (₹ crore → INR)",
            "period": period,
        },

        "Total Liabilities": {
            "value": yf_liabilities[0],
            "unit": "INR",
            "source": "Yahoo Finance annual balance sheet",
            "secondary": screener_liabilities,
            "secondary_source": "Screener (₹ crore → INR)",
            "period": period,
        },

        "Net Worth / Equity": {
            "value": yf_equity[0],
            "unit": "INR",
            "source": "Yahoo Finance annual balance sheet",
            "secondary": None,
            "secondary_source": "",
            "period": period,
        },

        "Retained Earnings": {
            "value": yf_retained_earnings[0],
            "unit": "INR",
            "source": "Yahoo Finance annual balance sheet",
            "secondary": None,
            "secondary_source": "",
            "period": period,
        },

        "Total Debt": {
            "value": yf_debt[0],
            "unit": "INR",
            "source": "Yahoo Finance annual balance sheet",
            "secondary": None,
            "secondary_source": "",
            "period": period,
        },

        "Cash & Equivalents": {
            "value": yf_cash[0],
            "unit": "INR",
            "source": "Yahoo Finance annual balance sheet",
            "secondary": None,
            "secondary_source": "",
            "period": period,
        },

        "Current Ratio": {
            "value": screener_current_ratio
            if screener_current_ratio is not None
            else yahoo_current_ratio,
            "unit": "x",
            "source": (
                "Screener"
                if screener_current_ratio is not None
                else "Yahoo-derived"
            ),
            "secondary": (
                yahoo_current_ratio
                if screener_current_ratio is not None
                else None
            ),
            "secondary_source": (
                "Yahoo-derived"
                if screener_current_ratio is not None
                else ""
            ),
            "period": period,
        },

        "Debt / Equity": {
            "value": screener_de if screener_de is not None else yahoo_de,
            "unit": "x",
            "source": (
                "Screener"
                if screener_de is not None
                else "Yahoo-derived"
            ),
            "secondary": (
                yahoo_de
                if screener_de is not None
                else None
            ),
            "secondary_source": (
                "Yahoo-derived"
                if screener_de is not None
                else ""
            ),
            "period": period,
        },

        "ROCE": {
            "value": screener_roce if screener_roce is not None else yahoo_roce,
            "unit": "%",
            "source": (
                "Screener"
                if screener_roce is not None
                else "Yahoo-derived"
            ),
            "secondary": (
                yahoo_roce
                if screener_roce is not None
                else None
            ),
            "secondary_source": (
                "Yahoo-derived"
                if screener_roce is not None
                else ""
            ),
            "period": period,
        },

        "Interest Coverage": {
            "value": yahoo_interest_coverage,
            "unit": "x",
            "source": "Yahoo-derived: EBIT / Interest Expense",
            "secondary": None,
            "secondary_source": "",
            "period": period,
        },

        "DSO / Debtor Days": {
            "value": screener_dso if screener_dso is not None else dso_derived,
            "unit": "days",
            "source": (
                "Screener"
                if screener_dso is not None
                else "Yahoo-derived"
            ),
            "secondary": (
                dso_derived if screener_dso is not None else None
            ),
            "secondary_source": (
                "Yahoo-derived"
                if screener_dso is not None
                else ""
            ),
            "period": period,
        },

        "DIO / Inventory Days": {
            "value": screener_dio if screener_dio is not None else dio_derived,
            "unit": "days",
            "source": (
                "Screener"
                if screener_dio is not None
                else "Yahoo-derived"
            ),
            "secondary": (
                dio_derived if screener_dio is not None else None
            ),
            "secondary_source": (
                "Yahoo-derived"
                if screener_dio is not None
                else ""
            ),
            "period": period,
        },

        "DPO / Creditor Days": {
            "value": screener_dpo if screener_dpo is not None else dpo_derived,
            "unit": "days",
            "source": (
                "Screener"
                if screener_dpo is not None
                else "Yahoo-derived"
            ),
            "secondary": (
                dpo_derived if screener_dpo is not None else None
            ),
            "secondary_source": (
                "Yahoo-derived"
                if screener_dpo is not None
                else ""
            ),
            "period": period,
        },

        "Promoter Pledging": {
            "value": screener_pledge,
            "unit": "%",
            "source": "Screener",
            "secondary": None,
            "secondary_source": "",
            "period": "Latest available",
        },
    }

    # ------------------------------------------------------------
    # Add a validation status to each metric.
    # ------------------------------------------------------------

    comparable_count = 0
    comparable_match = 0
    available_count = 0

    for name, m in metrics.items():
        value = safe_float(m["value"])
        secondary = safe_float(m["secondary"])

        if value is not None:
            available_count += 1

        diff = pct_difference(value, secondary)

        if diff is not None:
            comparable_count += 1

            # Ratios can differ because providers define them slightly
            # differently. A 5% tolerance is used for screening.
            if diff <= 5:
                comparable_match += 1
                m["validation"] = "✅ Cross-source match"
            elif diff <= 15:
                m["validation"] = "⚠️ Small discrepancy"
            else:
                m["validation"] = "🔴 Material discrepancy"

            m["difference_pct"] = diff

        elif value is not None:
            m["validation"] = "🟡 Single-source / derived"
            m["difference_pct"] = None

        else:
            m["validation"] = "❌ Missing"
            m["difference_pct"] = None

    # ------------------------------------------------------------
    # Core data-quality score
    # ------------------------------------------------------------

    core_metric_names = [
        "Revenue",
        "EBIT",
        "PAT / Net Income",
        "Operating Cash Flow",
        "Total Assets",
        "Total Liabilities",
        "Net Worth / Equity",
        "Total Debt",
        "Current Ratio",
        "Debt / Equity",
        "Interest Coverage",
        "ROCE",
    ]

    core_available = sum(
        metrics[n]["value"] is not None
        for n in core_metric_names
    )

    completeness = core_available / len(core_metric_names)

    agreement = (
        comparable_match / comparable_count
        if comparable_count
        else 0
    )

    # 60% completeness + 40% cross-source agreement.
    # It is explicitly a DATA-QUALITY SCORE, not a probability.
    quality_score = round(
        100 * (
            0.60 * completeness
            + 0.40 * agreement
        )
    )

    if quality_score >= 90:
        quality_band = "HIGH"
    elif quality_score >= 75:
        quality_band = "GOOD"
    elif quality_score >= 60:
        quality_band = "MODERATE"
    else:
        quality_band = "LOW"

    # ------------------------------------------------------------
    # Build audit table
    # ------------------------------------------------------------

    audit_rows = []

    for name, m in metrics.items():
        val = m["value"]
        sec = m["secondary"]

        def fmt(v):
            if v is None:
                return "—"
            if abs(v) >= 1e9:
                return f"{v/1e9:,.2f} bn"
            if abs(v) >= 1e7:
                return f"{v/1e7:,.2f} cr"
            return f"{v:,.2f}"

        diff_text = (
            f"{m['difference_pct']:.1f}%"
            if m["difference_pct"] is not None
            else "—"
        )

        audit_rows.append([
            name,
            fmt(val),
            m["unit"],
            m["period"],
            m["source"],
            (
                fmt(sec)
                if sec is not None
                else "—"
            ),
            m["secondary_source"] or "—",
            diff_text,
            m["validation"],
        ])

    audit_df = pd.DataFrame(
        audit_rows,
        columns=[
            "Metric",
            "Primary Value",
            "Unit",
            "Period",
            "Primary Source",
            "Cross-Check Value",
            "Cross-Check Source",
            "Difference",
            "Validation",
        ]
    )

    return {
        "symbol": symbol,
        "company_name": company_name,
        "sector": sector,
        "industry": industry,
        "is_financial": is_financial,
        "period": period,
        "metrics": metrics,
        "audit_df": audit_df,
        "quality_score": quality_score,
        "quality_band": quality_band,
        "screener_error": screener_error,
        "yahoo_error": yahoo_error,
        "yahoo": yahoo,
        "screener": screener,
    }


# ================================================================
# 8. FORMATTING HELPERS
# ================================================================

def display_number(value, unit=""):
    if value is None:
        return "Not available"

    if unit == "INR":
        v = abs(value)

        if v >= 1e11:
            text = f"₹{value/1e11:.2f} lakh cr"
        elif v >= 1e7:
            text = f"₹{value/1e7:.2f} cr"
        elif v >= 1e5:
            text = f"₹{value/1e5:.2f} lakh"
        else:
            text = f"₹{value:,.0f}"

        return text

    if unit == "x":
        return f"{value:.2f}x"

    if unit == "%":
        return f"{value:.2f}%"

    if unit == "days":
        return f"{value:.1f}d"

    return f"{value:.2f}"


def get_metric(dataset, name):
    try:
        return dataset["metrics"][name]["value"]
    except Exception:
        return None


# ================================================================
# 9. DISTRESS MODELS
# ================================================================

def calculate_altman(dataset):
    """
    Altman-style score for operating / non-financial companies.

    The model requires an actual retained-earnings figure. We deliberately
    do not substitute total equity for retained earnings because doing so
    changes the meaning of the Altman variable.
    """

    if dataset["is_financial"]:
        return {
            "available": False,
            "score": None,
            "status": "Not appropriate for financial institutions",
            "reason": "Use sector-specific bank/insurer distress metrics."
        }

    m = dataset["metrics"]

    assets = m["Total Assets"]["value"]
    liabilities = m["Total Liabilities"]["value"]
    retained_earnings = m["Retained Earnings"]["value"]
    ebit = m["EBIT"]["value"]
    revenue = m["Revenue"]["value"]

    yf_bs = (
        dataset["yahoo"]["balance_sheet"]
        if dataset["yahoo"]
        else pd.DataFrame()
    )

    current_assets = get_df_metric(
        yf_bs,
        BALANCE_ALIASES["current_assets"],
        0
    )

    current_liabilities = get_df_metric(
        yf_bs,
        BALANCE_ALIASES["current_liabilities"],
        0
    )

    required = [
        assets,
        liabilities,
        retained_earnings,
        ebit,
        revenue,
        current_assets,
        current_liabilities,
    ]

    if any(v is None for v in required):
        return {
            "available": False,
            "score": None,
            "status": "Insufficient validated inputs",
            "reason": "A required Altman input is unavailable."
        }

    if assets <= 0 or liabilities <= 0:
        return {
            "available": False,
            "score": None,
            "status": "Invalid denominator",
            "reason": "Total assets and liabilities must be positive."
        }

    market_cap = (
        dataset["yahoo"].get("market_cap_inr")
        if dataset["yahoo"]
        else None
    )

    if market_cap is None or market_cap <= 0:
        return {
            "available": False,
            "score": None,
            "status": "Insufficient validated inputs",
            "reason": "Market capitalization unavailable."
        }

    working_capital = current_assets - current_liabilities

    x1 = working_capital / assets
    x2 = retained_earnings / assets
    x3 = ebit / assets
    x4 = market_cap / liabilities
    x5 = revenue / assets

    z = (
        1.2 * x1
        + 1.4 * x2
        + 3.3 * x3
        + 0.6 * x4
        + 0.999 * x5
    )

    if z < 1.81:
        zone = "High-distress zone"
    elif z < 2.99:
        zone = "Grey zone"
    else:
        zone = "Lower-distress zone"

    return {
        "available": True,
        "score": round(z, 2),
        "status": zone,
        "reason": "Altman-style operating-company screen."
    }


def calculate_beneish(dataset):
    """
    Calculate the eight-variable Beneish M-Score only when the required
    two-year inputs are genuinely available.

    No hard-coded proxy values are used. Missing or invalid inputs cause the
    model to be marked unavailable instead of manufacturing a score.
    """

    yf_fin = (
        dataset["yahoo"]["financials"]
        if dataset["yahoo"]
        else pd.DataFrame()
    )

    yf_bs = (
        dataset["yahoo"]["balance_sheet"]
        if dataset["yahoo"]
        else pd.DataFrame()
    )

    yf_cf = (
        dataset["yahoo"]["cashflow"]
        if dataset["yahoo"]
        else pd.DataFrame()
    )

    revenue = get_series_values(
        yf_fin, FINANCIAL_ALIASES["revenue"], 2
    )

    gross_profit = get_series_values(
        yf_fin, FINANCIAL_ALIASES["gross_profit"], 2
    )

    cogs = get_series_values(
        yf_fin, FINANCIAL_ALIASES["cost_of_revenue"], 2
    )

    sga = get_series_values(
        yf_fin, FINANCIAL_ALIASES["sga"], 2
    )

    depreciation = get_series_values(
        yf_fin, FINANCIAL_ALIASES["depreciation"], 2
    )

    net_income = get_series_values(
        yf_fin, FINANCIAL_ALIASES["net_income"], 2
    )

    ar = get_series_values(
        yf_bs, BALANCE_ALIASES["accounts_receivable"], 2
    )

    current_assets = get_series_values(
        yf_bs, BALANCE_ALIASES["current_assets"], 2
    )

    current_liabilities = get_series_values(
        yf_bs, BALANCE_ALIASES["current_liabilities"], 2
    )

    assets = get_series_values(
        yf_bs, BALANCE_ALIASES["total_assets"], 2
    )

    ppe = get_series_values(
        yf_bs, BALANCE_ALIASES["net_ppe"], 2
    )

    long_debt = get_series_values(
        yf_bs, BALANCE_ALIASES["long_term_debt"], 2
    )

    cfo = get_series_values(
        yf_cf, CASHFLOW_ALIASES["operating_cash_flow"], 2
    )

    # Derive gross profit when revenue and COGS are present but gross profit
    # is not separately reported.
    for i in range(2):
        if (
            gross_profit[i] is None
            and revenue[i] is not None
            and cogs[i] is not None
        ):
            gross_profit[i] = revenue[i] - abs(cogs[i])

    # Expenses may be shown as negative values by statement providers.
    sga = [abs(v) if v is not None else None for v in sga]
    depreciation = [
        abs(v) if v is not None else None
        for v in depreciation
    ]

    required_arrays = [
        revenue,
        gross_profit,
        sga,
        depreciation,
        net_income,
        ar,
        current_assets,
        current_liabilities,
        assets,
        ppe,
        long_debt,
        cfo,
    ]

    if any(
        any(v is None for v in arr)
        for arr in required_arrays
    ):
        return {
            "available": False,
            "score": None,
            "status": "Full two-year Beneish inputs unavailable",
            "components": {}
        }

    if any(
        v == 0
        for arr in [revenue, gross_profit, assets]
        for v in arr
    ):
        return {
            "available": False,
            "score": None,
            "status": "Beneish inputs contain zero denominators",
            "components": {}
        }

    try:
        # Current year / prior year. The previous version inverted DSRI;
        # this implementation follows the standard current-to-prior direction.
        dsri = (
            (ar[0] / revenue[0])
            /
            (ar[1] / revenue[1])
        )

        # Gross Margin Index = prior gross margin / current gross margin.
        gmi = (
            (gross_profit[1] / revenue[1])
            /
            (gross_profit[0] / revenue[0])
        )

        # Asset Quality Index.
        aqi_current = 1 - (
            (current_assets[0] + ppe[0])
            / assets[0]
        )

        aqi_prior = 1 - (
            (current_assets[1] + ppe[1])
            / assets[1]
        )

        if aqi_prior == 0:
            raise ZeroDivisionError("AQI prior-period denominator is zero.")

        aqi = aqi_current / aqi_prior

        # Sales Growth Index.
        sgi = revenue[0] / revenue[1]

        # Depreciation Index = prior rate / current rate.
        prior_dep_den = depreciation[1] + ppe[1]
        current_dep_den = depreciation[0] + ppe[0]

        if prior_dep_den == 0 or current_dep_den == 0:
            raise ZeroDivisionError(
                "Depreciation-rate denominator is zero."
            )

        prior_dep_rate = depreciation[1] / prior_dep_den
        current_dep_rate = depreciation[0] / current_dep_den

        if current_dep_rate == 0:
            raise ZeroDivisionError(
                "Current depreciation-rate denominator is zero."
            )

        depi = prior_dep_rate / current_dep_rate

        # SG&A Index = current expense/revenue divided by prior expense/revenue.
        sgai = (
            (sga[0] / revenue[0])
            /
            (sga[1] / revenue[1])
            if sga[1] != 0
            else 1.0
        )

        # Leverage Index.
        current_leverage_base = (
            current_liabilities[0] + long_debt[0]
        ) / assets[0]

        prior_leverage_base = (
            current_liabilities[1] + long_debt[1]
        ) / assets[1]

        if prior_leverage_base == 0:
            raise ZeroDivisionError(
                "LVGI prior-period denominator is zero."
            )

        lvgi = current_leverage_base / prior_leverage_base

        # Total Accruals to Total Assets.
        tata = (
            net_income[0] - cfo[0]
        ) / assets[0]

        score = (
            -4.84
            + 0.92 * dsri
            + 0.528 * gmi
            + 0.404 * aqi
            + 0.892 * sgi
            + 0.115 * depi
            - 0.172 * sgai
            - 0.327 * lvgi
            + 4.679 * tata
        )

        status = (
            "Potential manipulation-risk screen flag"
            if score > -1.78
            else "Below conventional warning threshold"
        )

        return {
            "available": True,
            "score": round(score, 2),
            "status": status,
            "components": {
                "DSRI": round(dsri, 2),
                "GMI": round(gmi, 2),
                "AQI": round(aqi, 2),
                "SGI": round(sgi, 2),
                "DEPI": round(depi, 2),
                "SGAI": round(sgai, 2),
                "LVGI": round(lvgi, 2),
                "TATA": round(tata, 3),
            }
        }

    except Exception as e:
        return {
            "available": False,
            "score": None,
            "status": f"Calculation unavailable: {e}",
            "components": {}
        }


def calculate_quantitative(dataset, interest_shock_pct):
    m = dataset["metrics"]

    ic = m["Interest Coverage"]["value"]
    ccc = None

    dso = m["DSO / Debtor Days"]["value"]
    dio = m["DIO / Inventory Days"]["value"]
    dpo = m["DPO / Creditor Days"]["value"]

    if not dataset["is_financial"] and all(
        v is not None for v in [dso, dio, dpo]
    ):
        ccc = dso + dio - dpo

    stressed_ic = None

    if ic is not None:
        stressed_ic = ic / (
            1 + interest_shock_pct / 100
        )

    altman = calculate_altman(dataset)
    beneish = calculate_beneish(dataset)

    signals = []

    current_ratio = m["Current Ratio"]["value"]
    de = m["Debt / Equity"]["value"]
    ocf = m["Operating Cash Flow"]["value"]
    pledge = m["Promoter Pledging"]["value"]

    if current_ratio is not None and current_ratio < 1:
        signals.append(
            f"Current ratio below 1.0x ({current_ratio:.2f}x)"
        )

    if stressed_ic is not None and stressed_ic < 1.5:
        signals.append(
            f"Stressed interest coverage falls below 1.5x "
            f"({stressed_ic:.2f}x)"
        )

    if ocf is not None and ocf < 0:
        signals.append("Negative operating cash flow")

    if de is not None and de > 2:
        signals.append(
            f"High debt/equity ({de:.2f}x)"
        )

    if ccc is not None and ccc > 90:
        signals.append(
            f"Long cash conversion cycle ({ccc:.0f} days)"
        )

    if (
        altman["available"]
        and altman["score"] < 1.81
    ):
        signals.append(
            f"Altman-style score in high-distress zone "
            f"({altman['score']:.2f})"
        )

    if (
        beneish["available"]
        and beneish["score"] > -1.78
    ):
        signals.append(
            f"Beneish M-Score above -1.78 "
            f"({beneish['score']:.2f})"
        )

    if pledge is not None and pledge > 30:
        signals.append(
            f"High promoter pledging ({pledge:.1f}%)"
        )

    # Status.
    if len(signals) >= 4:
        health = "HIGH RISK"
    elif len(signals) >= 2:
        health = "MODERATE CONCERN"
    else:
        health = "STABLE / LOW RISK"

    return {
        "health": health,
        "signals": signals,
        "stressed_ic": stressed_ic,
        "ccc": ccc,
        "altman": altman,
        "beneish": beneish,
    }


# ================================================================
# 10. COMPANY FETCH HANDLER
# ================================================================

CURRENT_DATASET = {
    "symbol": None,
    "company_name": "No company loaded"
}


def fetch_company_data(dropdown_choice, custom_ticker):
    global CURRENT_DATASET

    if (
        dropdown_choice != "Custom / Other Indian Ticker"
        and dropdown_choice in NIFTY_50_MAP
    ):
        symbol = NIFTY_50_MAP[dropdown_choice]
    else:
        symbol = (
            custom_ticker
            .strip()
            .upper()
            .replace(".NS", "")
            .replace(".BO", "")
        )

    if not symbol:
        return (
            "⚠️ Please select a company or enter an Indian ticker.",
            "",
            audit_table_html(pd.DataFrame()),
            metrics_grid_html({})
        )

    try:
        dataset = build_company_dataset(symbol)
        CURRENT_DATASET = dataset

        q = dataset["quality_score"]
        band = dataset["quality_band"]

        source_notes = []

        if dataset["screener_error"]:
            source_notes.append(
                f"Screener: {dataset['screener_error']}"
            )

        if dataset["yahoo_error"]:
            source_notes.append(
                f"Yahoo Finance: {dataset['yahoo_error']}"
            )

        note_text = (
            "\n\n⚠️ Source notes:\n"
            + "\n".join(f"• {x}" for x in source_notes)
            if source_notes
            else ""
        )

        status = f"""
### {dataset["company_name"]}

**Ticker:** `{dataset["symbol"]}`  
**Sector:** `{dataset["sector"] or "Not available"}`  
**Industry:** `{dataset["industry"] or "Not available"}`  
**Financial statement period:** `{dataset["period"]}`

**Data Quality Score: {q}/100 — {band}**

This score measures **data coverage + cross-source agreement**. It is **not** a probability that the data is correct.

{note_text}
"""

        data_source_text = (
            "Screener → ratio/working-capital cross-check  •  "
            "Yahoo Finance → raw annual statements  •  "
            "Official annual report/NSE/BSE → final manual verification"
        )

        # Render ONLY cards whose validated value actually exists.
        # Missing metrics are omitted completely and the CSS grid
        # reflows the remaining cards automatically.
        return (
            status,
            data_source_text,
            audit_table_html(dataset["audit_df"]),
            metrics_grid_html(dataset["metrics"]),
        )

    except Exception as e:
        CURRENT_DATASET = {
            "symbol": symbol,
            "company_name": symbol
        }

        return (
            f"⚠️ **Could not load {symbol}.**\n\n{str(e)}",
            "",
            pd.DataFrame(),
            ""
        )


# ================================================================
# 11. ANALYSIS / REPORT
# ================================================================

def build_analysis_report(dataset, interest_shock_pct):
    q = calculate_quantitative(
        dataset,
        interest_shock_pct
    )

    m = dataset["metrics"]

    lines = []

    lines.append(
        f"## Overall Assessment: **{q['health']}**"
    )

    lines.append("")

    lines.append(
        "### Quantitative Evaluation"
    )

    lines.append("")

    def metric_text(metric_name, unit):
        metric = m.get(metric_name, {})
        value = metric.get("value") if isinstance(metric, dict) else None
        if value is None:
            return None
        return display_number(value, unit)

    liquidity_text = metric_text("Current Ratio", "x")
    leverage_text = metric_text("Debt / Equity", "x")
    roce_text = metric_text("ROCE", "%")
    ic_text = metric_text("Interest Coverage", "x")

    if liquidity_text is not None:
        lines.append(f"**Liquidity:** Current Ratio `{liquidity_text}`")

    if leverage_text is not None:
        lines.append(f"**Leverage:** Debt / Equity `{leverage_text}`")

    if roce_text is not None:
        lines.append(f"**Profitability:** ROCE `{roce_text}`")

    if ic_text is not None:
        lines.append(f"**Base Interest Coverage:** `{ic_text}`")

    stressed = q["stressed_ic"]

    if stressed is not None:
        lines.append(
            f"**Stressed Interest Coverage (+{interest_shock_pct}%):** "
            f"`{display_number(stressed, 'x')}`"
        )

    if q["ccc"] is not None:
        dso_v = m.get("DSO / Debtor Days", {}).get("value")
        dio_v = m.get("DIO / Inventory Days", {}).get("value")
        dpo_v = m.get("DPO / Creditor Days", {}).get("value")
        lines.append(
            f"**Cash Conversion Cycle:** `{q['ccc']:.1f} days` "
            f"(DSO {dso_v:.1f} + DIO {dio_v:.1f} − DPO {dpo_v:.1f})"
        )

    lines.append("")
    lines.append("### Distress-Model Diagnostics")
    lines.append("")

    if q["altman"]["available"]:
        lines.append(
            f"**Altman-style operating-company score:** "
            f"`{q['altman']['score']:.2f}` — {q['altman']['status']}"
        )
    else:
        lines.append(
            f"**Altman-style score:** `Not available` — "
            f"{q['altman']['reason']}"
        )

    if q["beneish"]["available"]:
        lines.append(
            f"**Beneish M-Score:** "
            f"`{q['beneish']['score']:.2f}` — {q['beneish']['status']}"
        )

        comp = q["beneish"]["components"]

        lines.append("")
        lines.append(
            "| Beneish Indicator | Value | Interpretation |"
        )
        lines.append("|---|---:|---|")

        indicator_names = {
            "DSRI": "Days Sales in Receivables Index (DSRI)",
            "GMI": "Gross Margin Index (GMI)",
            "AQI": "Asset Quality Index (AQI)",
            "SGI": "Sales Growth Index (SGI)",
            "DEPI": "Depreciation Index (DEPI)",
            "SGAI": "Sales, General & Administrative Expenses Index (SGAI)",
            "LVGI": "Leverage Index (LVGI)",
            "TATA": "Total Accruals to Total Assets (TATA)",
        }

        interpretations = {
            "DSRI": ">1 can increase concern when receivables rise faster relative to sales",
            "GMI": ">1 can indicate gross-margin deterioration",
            "AQI": ">1 can indicate a shift toward less directly operating assets",
            "SGI": ">1 indicates sales growth versus the prior year",
            "DEPI": ">1 can indicate slower depreciation relative to the prior year",
            "SGAI": ">1 can indicate selling, general & administrative expense pressure",
            "LVGI": ">1 indicates an increase in financial leverage",
            "TATA": "Higher values can indicate greater reliance on accruals rather than cash",
        }

        for k, v in comp.items():
            lines.append(
                f"| {indicator_names.get(k, k)} | `{v}` | {interpretations.get(k, '')} |"
            )
    else:
        lines.append(
            f"**Beneish M-Score:** `Not available` — "
            f"{q['beneish']['status']}"
        )

    lines.append("")
    lines.append("### Identified Risk Factors")
    lines.append("")

    if q["signals"]:
        for s in q["signals"]:
            lines.append(f"• {s}")
    else:
        lines.append(
            "• No major rule-based distress flags were triggered."
        )

    lines.append("")
    lines.append(
        "### Data Integrity Note"
    )
    lines.append(
        "Only extracted or independently derived values are used. "
        "Missing values are shown as unavailable rather than replaced "
        "with default assumptions. Final submission-quality figures "
        "should still be checked against the company's audited annual "
        "report and NSE/BSE filing."
    )

    return "\n".join(lines), q


# ================================================================
# 12. AI FUNCTIONS
# ================================================================

def make_ai_payload(dataset, q):
    m = dataset["metrics"]

    return {
        "company": dataset["company_name"],
        "ticker": dataset["symbol"],
        "sector": dataset["sector"],
        "industry": dataset["industry"],
        "period": dataset["period"],
        "data_quality_score": dataset["quality_score"],
        "data_quality_band": dataset["quality_band"],

        "revenue_inr": m["Revenue"]["value"],
        "ebit_inr": m["EBIT"]["value"],
        "pat_inr": m["PAT / Net Income"]["value"],
        "operating_cash_flow_inr": m["Operating Cash Flow"]["value"],
        "total_assets_inr": m["Total Assets"]["value"],
        "total_liabilities_inr": m["Total Liabilities"]["value"],
        "equity_inr": m["Net Worth / Equity"]["value"],
        "debt_inr": m["Total Debt"]["value"],
        "cash_inr": m["Cash & Equivalents"]["value"],

        "current_ratio": m["Current Ratio"]["value"],
        "debt_to_equity": m["Debt / Equity"]["value"],
        "interest_coverage": m["Interest Coverage"]["value"],
        "roce_pct": m["ROCE"]["value"],

        "dso": m["DSO / Debtor Days"]["value"],
        "dio": m["DIO / Inventory Days"]["value"],
        "dpo": m["DPO / Creditor Days"]["value"],

        "stressed_interest_coverage": q["stressed_ic"],
        "cash_conversion_cycle": q["ccc"],

        "altman_style_score": (
            q["altman"]["score"]
            if q["altman"]["available"]
            else None
        ),

        "beneish_m_score": (
            q["beneish"]["score"]
            if q["beneish"]["available"]
            else None
        ),

        "risk_classification": q["health"],
        "risk_signals": q["signals"],
    }


def generate_ai_insight(
    interest_shock_pct,
):
    global CURRENT_DATASET

    if not CURRENT_DATASET or not CURRENT_DATASET.get("symbol"):
        return "⚠️ Extract company data first."

    dataset = CURRENT_DATASET

    report, q = build_analysis_report(
        dataset,
        interest_shock_pct
    )

    payload = make_ai_payload(
        dataset,
        q
    )

    prompt = f"""
You are the senior credit analyst inside a financial distress
screening dashboard.

IMPORTANT:
The numbers below have already passed the dashboard's data-validation
layer. Do NOT replace them, invent additional numbers, or assume missing
values.

Validated analytical dataset:
{json.dumps(payload, indent=2, default=str)}

Write a concise executive credit insight in 5-7 sentences.

Cover:
1. Overall financial condition.
2. Liquidity.
3. Leverage and debt sustainability.
4. Interest-rate stress.
5. Earnings-quality evidence ONLY if Beneish is available.
6. The most important risk signal.
7. A balanced conclusion.

Explicitly acknowledge data-quality limitations when the score is below 90.

Do not call a company "likely to default" solely from these screening models.
Do not provide personalized investment advice.
"""

    return call_gemini_api(prompt)


def chat_response(
    message,
    history,
    interest_shock_pct,
):
    global CURRENT_DATASET

    if not CURRENT_DATASET or not CURRENT_DATASET.get("symbol"):
        return "Please extract a company first."

    dataset = CURRENT_DATASET

    _, q = build_analysis_report(
        dataset,
        interest_shock_pct
    )

    payload = make_ai_payload(
        dataset,
        q
    )

    prompt = f"""
You are the AI Advisory Assistant for an institutional distress
screening engine.

Company:
{dataset["company_name"]}

Validated dataset:
{json.dumps(payload, indent=2, default=str)}

User question:
{message}

Answer directly and professionally.

Rules:
- Use the supplied validated numbers.
- Never invent missing data.
- Explain the financial logic.
- Mention when a model is unavailable or not applicable.
- Distinguish a screening flag from proof of distress.
- Do not provide personalized investment advice.
"""

    return call_gemini_api(prompt)


# ================================================================
# 13. UI HELPERS
# ================================================================

def metric_html(label, value, subtext=""):
    """Render one dashboard metric card."""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{html.escape(label)}</div>
        <div class="metric-value">{html.escape(str(value))}</div>
        <div class="metric-sub">{html.escape(subtext)}</div>
    </div>
    """


def metrics_grid_html(metrics):
    """
    Render ONLY metrics that actually have validated data.

    Missing metrics are omitted completely rather than displayed as
    dashes. The CSS grid automatically reflows the remaining cards,
    so there are no blank holes in the dashboard.
    """

    card_specs = [
        ("Revenue", "Revenue", "Latest validated annual", "INR"),
        ("EBIT", "EBIT", "Earnings before interest & tax", "INR"),
        ("PAT / Net Income", "PAT / Net Income", "Latest validated annual", "INR"),
        ("Operating Cash Flow", "Operating Cash Flow", "Cash generated from operations", "INR"),
        ("Current Ratio", "Current Ratio", "Current assets / current liabilities", "x"),
        ("Debt / Equity", "Debt / Equity", "Leverage", "x"),
        ("Interest Coverage", "Interest Coverage", "EBIT / interest expense", "x"),
        ("ROCE", "ROCE", "Return on capital employed", "%"),
    ]

    cards = []

    for _, metric_name, subtext, unit in card_specs:
        metric = metrics.get(metric_name)

        if not metric:
            continue

        value = metric.get("value")

        # IMPORTANT: no placeholder card for missing data.
        if value is None:
            continue

        cards.append(
            metric_html(
                metric_name,
                display_number(value, unit),
                subtext
            )
        )

    if not cards:
        return """
        <div class="metrics-empty">
            No validated dashboard metrics are currently available.
            Extract a company or review the Data Verification section.
        </div>
        """

    return '<div class="metric-grid">' + ''.join(cards) + '</div>'



def audit_table_html(df):
    """Render the verification table as stable HTML, independent of Gradio's table DOM/theme."""

    columns = [
        "Metric",
        "Primary Value",
        "Unit",
        "Period",
        "Primary Source",
        "Cross-Check Value",
        "Cross-Check Source",
        "Difference",
        "Validation",
    ]

    if df is None or df.empty:
        return """
        <div class="audit-empty">
            No verification records yet. Extract a company to build the audit trail.
        </div>
        """

    def esc(v):
        return html.escape("" if v is None else str(v))

    rows = []

    for _, row in df.iterrows():
        validation = str(row.get("Validation", ""))

        if "Cross-source match" in validation:
            status_class = "ok"
        elif "Material discrepancy" in validation:
            status_class = "bad"
        elif "Missing" in validation:
            status_class = "missing"
        else:
            status_class = "warn"

        cells = []

        for col in columns:
            value = row.get(col, "")

            if value in ["—", "-", None, ""]:
                value = "Not available"

            cells.append(f"<td>{esc(value)}</td>")

        cells[-1] = (
            f'<td><span class="audit-status {status_class}">'
            f'{esc(validation)}</span></td>'
        )

        rows.append(
            "<tr>" + "".join(cells) + "</tr>"
        )

    return """
    <div class="audit-table-scroll">
        <table class="audit-table-html">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Primary Value</th>
                    <th>Unit</th>
                    <th>Period</th>
                    <th>Primary Source</th>
                    <th>Cross-Check Value</th>
                    <th>Cross-Check Source</th>
                    <th>Difference</th>
                    <th>Validation</th>
                </tr>
            </thead>
            <tbody>
    """ + "".join(rows) + """
            </tbody>
        </table>
    </div>
    """


def render_chat_html(history):
    """Render chat messages with completely controlled light styling."""

    if not history:
        return """
        <div class="chat-empty">
            <div class="chat-empty-title">Your advisory desk is ready.</div>
            <div class="chat-empty-sub">
                Ask about liquidity, leverage, cash flow, distress signals,
                stress testing or the model outputs for the loaded company.
            </div>
        </div>
        """

    blocks = []

    for role, message in history:
        safe_message = html.escape(str(message)).replace("\n", "<br>")

        if role == "user":
            blocks.append(
                f"""
                <div class="chat-row user-row">
                    <div class="chat-bubble user-bubble">
                        <div class="chat-role">You</div>
                        <div>{safe_message}</div>
                    </div>
                </div>
                """
            )
        else:
            blocks.append(
                f"""
                <div class="chat-row assistant-row">
                    <div class="chat-bubble assistant-bubble">
                        <div class="chat-role">AI Advisory Assistant</div>
                        <div>{safe_message}</div>
                    </div>
                </div>
                """
            )

    return '<div class="chat-transcript">' + "".join(blocks) + "</div>"


def submit_chat(message, history, interest_shock_pct):
    history = history or []

    if not message or not message.strip():
        return "", history, render_chat_html(history)

    user_message = message.strip()

    try:
        answer = chat_response(
            user_message,
            interest_shock_pct
        )
    except Exception as e:
        answer = (
            "⚠️ The advisory assistant encountered an error: "
            + str(e)
        )

    history = history + [
        ("user", user_message),
        ("assistant", answer),
    ]

    return "", history, render_chat_html(history)


def clear_chat():
    return [], render_chat_html([])


# ================================================================
# 14. EDITORIAL / SCRAPBOOK CSS
# ================================================================

custom_css = r"""
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600;700&family=DM+Sans:wght@400;500;600&display=swap');

:root {
    --blue: #a8bfcb;
    --blue-dark: #7896a4;
    --cream: #f8f1e5;
    --paper: #fffdf8;
    --ink: #292827;
    --muted: #716c67;
    --line: #dfd4c5;
    --accent: #b29a80;
    --white: #ffffff;
}

/* ============================================================
   LIGHT-MODE LOCK
   Prevent Gradio's runtime theme from turning cream panels,
   labels, or inputs into dark-mode elements.
   ============================================================ */

html,
body {
    background: #a8bfcb !important;
    color: #292827 !important;
    color-scheme: light !important;
}

.gradio-container {
    background: #a8bfcb !important;
    color: #292827 !important;
    color-scheme: light !important;
    max-width: 1120px !important;
    margin: 0 auto !important;
    padding: 0 18px 50px 18px !important;

    /* Stabilize Gradio's theme variables */
    --body-text-color: #292827 !important;
    --body-text-color-subdued: #716c67 !important;
    --block-label-text-color: #504b47 !important;
    --input-text-color: #292827 !important;
    --input-background-fill: #fffdfa !important;
    --block-background-fill: #fffdf8 !important;
    --panel-background-fill: #fffdf8 !important;
    --block-border-color: #dfd4c5 !important;
    --border-color-primary: #dfd4c5 !important;
}

/* Force textual elements to remain readable even if a Gradio
   light/dark class is applied after the app is constructed. */
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4,
.gradio-container h5,
.gradio-container h6,
.gradio-container p,
.gradio-container label,
.gradio-container .prose,
.gradio-container .prose p,
.gradio-container .prose li,
.gradio-container .prose strong,
.gradio-container .prose em,
.gradio-container .section-title,
.gradio-container .section-subtitle,
.gradio-container .hero-kicker,
.gradio-container .hero-title,
.gradio-container .hero-sub,
.gradio-container .step-title,
.gradio-container .step-text,
.gradio-container .metric-label,
.gradio-container .metric-value,
.gradio-container .metric-sub,
.gradio-container .disclaimer {
    color: #292827 !important;
}

.gradio-container .section-subtitle,
.gradio-container .hero-kicker,
.gradio-container .hero-sub,
.gradio-container .step-text,
.gradio-container .metric-sub,
.gradio-container .disclaimer {
    color: #716c67 !important;
}

/* ============================================================
   MAIN PAPER / SCRAPBOOK LAYOUT
   ============================================================ */

.gradio-container {
    font-family: 'DM Sans', Arial, sans-serif !important;
}

h1, h2, h3, h4, h5, h6,
.hero-title,
.section-title,
.step-number,
.metric-value {
    font-family: 'Cormorant Garamond', Georgia, serif !important;
}

.paper-section {
    background: #fffdf8 !important;
    color: #292827 !important;
    border: 1px solid rgba(70,60,50,.08) !important;
    border-radius: 2px !important;
    padding: 34px 36px !important;
    margin: 20px 0 !important;
    box-shadow: 0 8px 24px rgba(65,74,78,.08) !important;
}

.blue-section {
    background: #a8bfcb !important;
    color: #292827 !important;
    padding: 28px 6px !important;
}

.hero {
    position: relative !important;
    background: #fffdf8 !important;
    color: #292827 !important;
    border-radius: 2px !important;
    overflow: hidden !important;
    margin-top: 22px !important;
    box-shadow: 0 10px 26px rgba(65,74,78,.10) !important;
}

.hero::before,
.hero::after {
    content: "" !important;
    display: block !important;
    height: 25px !important;
    background: repeating-linear-gradient(
        90deg,
        #d8e7ea 0px,
        #d8e7ea 18px,
        #f8f1e5 18px,
        #f8f1e5 34px
    ) !important;
}

.hero-inner {
    padding: 38px 44px 34px 44px !important;
    background: #fffdf8 !important;
}

.hero-kicker {
    text-transform: uppercase !important;
    letter-spacing: .20em !important;
    font-size: .72rem !important;
    text-align: center !important;
    margin-bottom: 10px !important;
}

.hero-title {
    font-size: clamp(2.7rem, 6vw, 4.4rem) !important;
    line-height: .90 !important;
    text-align: center !important;
    font-weight: 600 !important;
    margin: 4px 0 10px 0 !important;
}

.hero-sub {
    max-width: 760px !important;
    margin: 0 auto !important;
    font-size: .93rem !important;
    line-height: 1.6 !important;
    text-align: center !important;
}

.section-title {
    font-size: 2.1rem !important;
    text-align: center !important;
    margin: 0 0 5px 0 !important;
    font-weight: 600 !important;
}

.section-subtitle {
    text-align: center !important;
    font-size: .83rem !important;
    margin-bottom: 18px !important;
    line-height: 1.45 !important;
}


/* ============================================================
   OUTPUT SURFACES — keep all app-generated text readable
   ============================================================ */

.status-output,
.source-output,
.report-output {
    background: transparent !important;
    color: #292827 !important;
    -webkit-text-fill-color: #292827 !important;
}

.status-output * ,
.source-output * ,
.report-output * {
    color: #292827 !important;
    -webkit-text-fill-color: #292827 !important;
}

.status-output a,
.source-output a,
.report-output a {
    color: #536f7b !important;
    -webkit-text-fill-color: #536f7b !important;
}

.status-output .prose,
.source-output .prose,
.report-output .prose,
.status-output .prose * ,
.source-output .prose * ,
.report-output .prose * {
    color: #292827 !important;
    -webkit-text-fill-color: #292827 !important;
}

/* Markdown backticks must remain light and readable. Gradio's dark-mode
   defaults otherwise render inline code as near-black pills. */
.status-output code,
.source-output code,
.report-output code,
.ai-output-panel code,
.status-output pre,
.source-output pre,
.report-output pre,
.ai-output-panel pre {
    background: #f2eadf !important;
    color: #292827 !important;
    -webkit-text-fill-color: #292827 !important;
    border: 1px solid #dfd4c5 !important;
    border-radius: 5px !important;
    padding: 2px 6px !important;
    box-shadow: none !important;
}

.report-output table {
    width: 100% !important;
    border-collapse: collapse !important;
    background: #fffdfa !important;
    color: #292827 !important;
    margin: 14px 0 !important;
    font-size: .80rem !important;
}

.report-output th {
    background: #dce9ec !important;
    color: #2e3435 !important;
    -webkit-text-fill-color: #2e3435 !important;
    font-weight: 700 !important;
    padding: 10px 11px !important;
    border: 1px solid #c7d6da !important;
    text-align: left !important;
}

.report-output td {
    background: #fffdfa !important;
    color: #3b3835 !important;
    -webkit-text-fill-color: #3b3835 !important;
    padding: 10px 11px !important;
    border: 1px solid #e6ddd1 !important;
    vertical-align: top !important;
    line-height: 1.4 !important;
}

.report-output tr:nth-child(even) td {
    background: #fbf6ee !important;
}

/* ============================================================
   GRADIO CONTROL TEXT — prevents theme colour inheritance
   ============================================================ */

.gradio-container button,
.gradio-container button span,
.gradio-container [role="button"],
.gradio-container [role="button"] span,
.gradio-container [role="combobox"],
.gradio-container [role="combobox"] *,
.gradio-container [role="option"],
.gradio-container [role="option"] * {
    color: #292827 !important;
    -webkit-text-fill-color: #292827 !important;
}

.gradio-container svg {
    color: #5c5752 !important;
}

.gradio-container .editorial-btn,
.gradio-container .editorial-btn span,
.gradio-container .soft-blue-btn,
.gradio-container .soft-blue-btn span,
.gradio-container .chat-send,
.gradio-container .chat-send span,
.gradio-container .chat-clear,
.gradio-container .chat-clear span {
    -webkit-text-fill-color: inherit !important;
}


/* ============================================================
   INPUT AREA — force cream backgrounds instead of Gradio dark
   ============================================================ */

.form-card .label-wrap,
.form-card .label-wrap label,
.form-card label,
.form-card small,
.form-card p {
    color: #504b47 !important;
    -webkit-text-fill-color: #504b47 !important;
}

.form-card {
    background: #fbf7ef !important;
    color: #292827 !important;
    border: 1px solid #dfd4c5 !important;
    border-radius: 12px !important;
    padding: 20px !important;
}

.form-card,
.form-card > div,
.form-card .form,
.form-card .wrap,
.form-card .block {
    background-color: #fbf7ef !important;
    color: #292827 !important;
}

/* Textboxes, dropdowns, inputs, and their internal wrappers */
.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container .gr-input,
.gradio-container [role="combobox"],
.gradio-container [role="textbox"] {
    background: #fffdfa !important;
    color: #292827 !important;
    -webkit-text-fill-color: #292827 !important;
    border-color: #ddd2c4 !important;
    box-shadow: none !important;
}

.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
    color: #817a73 !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #817a73 !important;
}

.gradio-container label,
.gradio-container .label-wrap,
.gradio-container .wrap label {
    color: #504b47 !important;
    -webkit-text-fill-color: #504b47 !important;
}

/* Dropdown button and menu */
.gradio-container [role="combobox"] * {
    color: #292827 !important;
}

.gradio-container [role="listbox"],
.gradio-container [role="option"] {
    background: #fffdfa !important;
    color: #292827 !important;
}

/* ============================================================
   WORKFLOW CARDS
   ============================================================ */

.step-card {
    background: rgba(255,253,248,.98) !important;
    color: #292827 !important;
    border: 1px solid rgba(70,60,50,.10) !important;
    border-radius: 10px !important;
    padding: 17px 15px !important;
    min-height: 108px !important;
    box-shadow: 0 5px 14px rgba(60,70,75,.06) !important;
}

.step-number {
    font-size: 1.35rem !important;
    color: #7896a4 !important;
    margin-bottom: 5px !important;
}

.step-title {
    font-weight: 600 !important;
    margin-bottom: 5px !important;
}

.step-text {
    font-size: .76rem !important;
    line-height: 1.4 !important;
}

/* ============================================================
   METRIC CARDS
   ============================================================ */

.metric-grid {
    display: grid !important;
    grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
    gap: 12px !important;
    margin: 14px 0 18px 0 !important;
    align-items: stretch !important;
}

.metric-card {
    background: #fffdf8 !important;
    color: #292827 !important;
    border: 1px solid #dfd4c5 !important;
    border-radius: 10px !important;
    padding: 16px 15px !important;
    min-height: 116px !important;
    box-sizing: border-box !important;
    text-align: left !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: flex-start !important;
}

.metric-label {
    font-size: .82rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: .08em !important;
    color: #5f5a55 !important;
    margin-bottom: 10px !important;
    line-height: 1.25 !important;
}

.metric-value {
    font-size: 1.78rem !important;
    line-height: 1 !important;
    font-weight: 600 !important;
    color: #292827 !important;
    margin-bottom: 7px !important;
    overflow-wrap: anywhere !important;
}

.metric-sub {
    font-size: .70rem !important;
    color: #716c67 !important;
    line-height: 1.3 !important;
}

.metrics-empty {
    border: 1px dashed #dfd4c5 !important;
    border-radius: 10px !important;
    padding: 26px 20px !important;
    text-align: center !important;
    color: #716c67 !important;
    background: rgba(255,253,248,.72) !important;
    font-size: .82rem !important;
}

/* ============================================================
   AUDIT TABLE
   ============================================================ */

.audit-wrap {
    border: 1px solid #dfd4c5 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    background: #fffdfa !important;
    color: #292827 !important;
}

.gradio-container .table-wrap,
.gradio-container .table,
.gradio-container table,
.gradio-container th,
.gradio-container td {
    color: #292827 !important;
    background: #fffdfa !important;
}


/* ============================================================
   STABLE CUSTOM AUDIT TABLE
   ============================================================ */

.audit-table-scroll {
    width: 100% !important;
    max-height: 560px !important;
    overflow: auto !important;
    border: 1px solid #dfd4c5 !important;
    border-radius: 10px !important;
    background: #fffdfa !important;
}

.audit-table-html {
    width: 100% !important;
    min-width: 1100px !important;
    border-collapse: separate !important;
    border-spacing: 0 !important;
    background: #fffdfa !important;
    color: #292827 !important;
    font-family: 'DM Sans', Arial, sans-serif !important;
    font-size: .76rem !important;
}

.audit-table-html th {
    position: sticky !important;
    top: 0 !important;
    z-index: 3 !important;
    background: #dce9ec !important;
    color: #2e3435 !important;
    -webkit-text-fill-color: #2e3435 !important;
    font-weight: 700 !important;
    text-align: left !important;
    padding: 12px 11px !important;
    border-bottom: 1px solid #c7d6da !important;
    white-space: nowrap !important;
}

.audit-table-html td {
    background: #fffdfa !important;
    color: #3b3835 !important;
    -webkit-text-fill-color: #3b3835 !important;
    padding: 11px !important;
    border-bottom: 1px solid #eee5d9 !important;
    vertical-align: top !important;
    line-height: 1.35 !important;
}

.audit-table-html tbody tr:nth-child(even) td {
    background: #fbf6ee !important;
}

.audit-table-html tbody tr:hover td {
    background: #f3eee5 !important;
}

.audit-status {
    display: inline-block !important;
    padding: 4px 8px !important;
    border-radius: 999px !important;
    font-size: .67rem !important;
    font-weight: 700 !important;
    white-space: nowrap !important;
}

.audit-status.ok {
    background: #e5efdf !important;
    color: #44603e !important;
}

.audit-status.warn {
    background: #f3ead7 !important;
    color: #735b2f !important;
}

.audit-status.bad {
    background: #f2dedd !important;
    color: #7c4440 !important;
}

.audit-status.missing {
    background: #ece8e2 !important;
    color: #6d675f !important;
}

.audit-empty {
    padding: 30px 20px !important;
    text-align: center !important;
    border: 1px dashed #dfd4c5 !important;
    border-radius: 10px !important;
    background: #fffdfa !important;
    color: #716c67 !important;
    font-size: .82rem !important;
}

/* ============================================================
   AI OUTPUT + CUSTOM CHAT
   ============================================================ */

.ai-output-panel {
    background: #fffdfa !important;
    border: 1px solid #dfd4c5 !important;
    border-radius: 12px !important;
    padding: 18px 20px !important;
    min-height: 120px !important;
    color: #292827 !important;
    -webkit-text-fill-color: #292827 !important;
    line-height: 1.6 !important;
}

.ai-output-panel,
.ai-output-panel *,
.ai-output-panel p,
.ai-output-panel li,
.ai-output-panel strong,
.ai-output-panel em,
.ai-output-panel h1,
.ai-output-panel h2,
.ai-output-panel h3 {
    color: #292827 !important;
    -webkit-text-fill-color: #292827 !important;
}

.chat-shell {
    background: #fbf7ef !important;
    border: 1px solid #dfd4c5 !important;
    border-radius: 12px !important;
    min-height: 330px !important;
    max-height: 520px !important;
    overflow: auto !important;
    padding: 15px !important;
    color: #292827 !important;
}

.chat-transcript {
    display: flex !important;
    flex-direction: column !important;
    gap: 12px !important;
}

.chat-row {
    display: flex !important;
    width: 100% !important;
}

.user-row { justify-content: flex-end !important; }
.assistant-row { justify-content: flex-start !important; }

.chat-bubble {
    max-width: 86% !important;
    border-radius: 14px !important;
    padding: 12px 14px !important;
    font-size: .82rem !important;
    line-height: 1.52 !important;
    box-sizing: border-box !important;
    color: #292827 !important;
    -webkit-text-fill-color: #292827 !important;
}

.user-bubble {
    background: #dbe8eb !important;
    border: 1px solid #bfd2d8 !important;
}

.assistant-bubble {
    background: #fffdfa !important;
    border: 1px solid #dfd4c5 !important;
    box-shadow: 0 3px 10px rgba(60,70,75,.04) !important;
}

.chat-role {
    font-size: .67rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: .08em !important;
    color: #6e6962 !important;
    margin-bottom: 5px !important;
}

.chat-empty {
    min-height: 290px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    align-items: center !important;
    text-align: center !important;
    padding: 30px !important;
}

.chat-empty-title {
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    font-size: 1.75rem !important;
    color: #292827 !important;
    margin-bottom: 8px !important;
}

.chat-empty-sub {
    max-width: 560px !important;
    font-size: .78rem !important;
    line-height: 1.5 !important;
    color: #716c67 !important;
}

.chat-input-row {
    background: #fbf7ef !important;
    border: 1px solid #dfd4c5 !important;
    border-radius: 12px !important;
    padding: 10px !important;
    margin-top: 10px !important;
}

.chat-input-row input,
.chat-input-row textarea {
    background: #fffdfa !important;
    color: #292827 !important;
    -webkit-text-fill-color: #292827 !important;
    border: 1px solid #ddd2c4 !important;
}

.chat-clear {
    background: #eee6dc !important;
    color: #4b4540 !important;
    -webkit-text-fill-color: #4b4540 !important;
    border: 1px solid #d8cbbb !important;
}

.chat-send {
    background: #d8c9ba !important;
    color: #2e2a27 !important;
    -webkit-text-fill-color: #2e2a27 !important;
    border: 1px solid #c5b3a2 !important;
}

/* ============================================================
   BUTTONS
   ============================================================ */

button,
.gr-button {
    border-radius: 999px !important;
    font-family: 'DM Sans', Arial, sans-serif !important;
}

.editorial-btn {
    background: #d8c9ba !important;
    color: #2e2a27 !important;
    -webkit-text-fill-color: #2e2a27 !important;
    border: 1px solid #c5b3a2 !important;
    font-weight: 600 !important;
    letter-spacing: .06em !important;
    text-transform: uppercase !important;
}

.editorial-btn:hover,
.editorial-btn:hover span {
    background: #2f2b28 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

.soft-blue-btn {
    background: #d9e6ea !important;
    color: #314148 !important;
    -webkit-text-fill-color: #314148 !important;
    border: 1px solid #b8cdd5 !important;
}

/* ============================================================
   SLIDER / RANGE INPUT
   ============================================================ */

.gradio-container input[type="range"] {
    background: transparent !important;
    color: #292827 !important;
}

.gradio-container .range,
.gradio-container .slider-container {
    background: #fffdfa !important;
    color: #292827 !important;
}

/* ============================================================
   DISCLAIMER
   ============================================================ */

.disclaimer {
    background: rgba(255,253,248,.88) !important;
    border-left: 3px solid #c3aa90 !important;
    padding: 14px 16px !important;
    border-radius: 4px !important;
    font-size: .76rem !important;
    line-height: 1.5 !important;
}

/* ============================================================
   RESPONSIVE GRID
   ============================================================ */

@media (max-width: 820px) {
    .metric-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    }

    .hero-inner,
    .paper-section {
        padding: 24px 20px !important;
    }
}

@media (max-width: 520px) {
    .metric-grid {
        grid-template-columns: 1fr !important;
    }
}

footer {
    display: none !important;
}
"""


# ================================================================
# 15. BUILD UI
# ================================================================

# Close an older Gradio app if the user re-runs this cell in the
# same Colab runtime. This avoids stale servers/components lingering
# between notebook runs.
_previous_demo = globals().get("demo")
if _previous_demo is not None:
    try:
        _previous_demo.close()
    except Exception:
        pass

with gr.Blocks(
    theme=gr.themes.Base(),
    css=custom_css,
    title="Institutional Distress Intelligence Engine"
) as demo:

    # ------------------------------------------------------------
    # HERO / BINDER-STYLE HEADER
    # ------------------------------------------------------------

    with gr.Column(elem_classes=["hero"]):

        gr.HTML("""
        <div class="hero-inner">
            <div class="hero-kicker">
                Institutional Credit Analytics
            </div>

            <div class="hero-title">
                Distress Intelligence Engine
            </div>

            <div class="hero-sub">
                Validated financial extraction, cross-source data checks,
                multi-model distress screening and macro stress testing —
                designed to make the numbers auditable before the AI speaks.
            </div>
        </div>
        """)

    # ------------------------------------------------------------
    # COMPANY INPUT
    # ------------------------------------------------------------

    with gr.Column(elem_classes=["paper-section"]):

        gr.Markdown(
            "### Select the enterprise",
            elem_classes=["section-title"]
        )

        gr.Markdown(
            "Load the latest available annual data, then inspect the audit trail before analysing.",
            elem_classes=["section-subtitle"]
        )

        with gr.Row(elem_classes=["form-card"]):

            with gr.Column(scale=2):

                nifty_dropdown = gr.Dropdown(
                    choices=list(NIFTY_50_MAP.keys()),
                    value="Tata Steel (TATASTEEL)",
                    label="Nifty 50 / Indian Enterprise"
                )

            with gr.Column(scale=1):

                custom_input = gr.Textbox(
                    label="Custom Indian Ticker",
                    placeholder="e.g. RELIANCE, SUZLON"
                )

            with gr.Column(scale=1):

                fetch_btn = gr.Button(
                    "01 · Extract & Validate",
                    elem_classes=["editorial-btn"]
                )

        status_box = gr.Markdown(
            "Choose a company and click **Extract & Validate**.",
            elem_classes=["status-output"]
        )

        source_box = gr.Markdown(
            "",
            elem_classes=["source-output"]
        )

    # ------------------------------------------------------------
    # WORKFLOW STRIP
    # ------------------------------------------------------------

    with gr.Column(elem_classes=["blue-section"]):

        gr.Markdown(
            "### How the engine works",
            elem_classes=["section-title"]
        )

        with gr.Row():

            gr.HTML("""
            <div class="step-card">
                <div class="step-number">01</div>
                <div class="step-title">Extract</div>
                <div class="step-text">Pull ratios and annual statements from two independent data sources.</div>
            </div>
            """)

            gr.HTML("""
            <div class="step-card">
                <div class="step-number">02</div>
                <div class="step-title">Validate</div>
                <div class="step-text">Compare comparable values and flag missing or materially different inputs.</div>
            </div>
            """)

            gr.HTML("""
            <div class="step-card">
                <div class="step-number">03</div>
                <div class="step-title">Stress-test</div>
                <div class="step-text">Run liquidity, leverage, working-capital and distress-model diagnostics.</div>
            </div>
            """)

            gr.HTML("""
            <div class="step-card">
                <div class="step-number">04</div>
                <div class="step-title">Interpret</div>
                <div class="step-text">Only after validation, Gemini converts the analytics into an executive credit view.</div>
            </div>
            """)

    # ------------------------------------------------------------
    # DATA AUDIT
    # ------------------------------------------------------------

    with gr.Column(elem_classes=["paper-section"]):

        gr.Markdown(
            "### Data Verification",
            elem_classes=["section-title"]
        )

        gr.Markdown(
            "The audit table shows the exact input, period, source and cross-check status used by the engine.",
            elem_classes=["section-subtitle"]
        )

        audit_table = gr.HTML(
            audit_table_html(pd.DataFrame()),
            elem_classes=["audit-wrap"]
        )


    # ------------------------------------------------------------
    # QUANTITATIVE EVALUATION
    # ------------------------------------------------------------

    with gr.Column(elem_classes=["paper-section"]):

        gr.Markdown(
            "### Quantitative Evaluation",
            elem_classes=["section-title"]
        )

        gr.Markdown(
            "Aligned summary cards first; detailed diagnostics underneath.",
            elem_classes=["section-subtitle"]
        )

        # One dynamic grid instead of fixed card components.
        # This is important: unavailable data is omitted entirely,
        # and the remaining cards automatically re-align into the grid.
        metrics_grid = gr.HTML(
            metrics_grid_html({})
        )

        # --------------------------------------------------------
        # Macro stress control
        # --------------------------------------------------------

        with gr.Row(elem_classes=["form-card"]):

            interest_shock = gr.Slider(
                minimum=0,
                maximum=100,
                value=25,
                step=5,
                label="Macro Interest-Rate Shock (+%)"
            )

            analyze_btn = gr.Button(
                "02 · Run Validated Distress Analysis",
                elem_classes=["editorial-btn"]
            )

        report_output = gr.Markdown(
            "Run the analysis after extracting a company.",
            elem_classes=["report-output"]
        )

    # ------------------------------------------------------------
    # AI INSIGHT
    # ------------------------------------------------------------

    with gr.Column(elem_classes=["blue-section"]):

        gr.Markdown(
            "### Executive Credit Insight",
            elem_classes=["section-title"]
        )

        gr.Markdown(
            "Gemini is used for interpretation only. It receives the validated analytical dataset and is not allowed to invent missing figures.",
            elem_classes=["section-subtitle"]
        )

        ai_btn = gr.Button(
            "03 · Generate AI Credit Insight",
            elem_classes=["soft-blue-btn"]
        )

        ai_output = gr.Markdown(
            "Generate the executive insight after extracting and analysing a company.",
            elem_classes=["ai-output-panel"]
        )

    # ------------------------------------------------------------
    # CHATBOT — THEME-INDEPENDENT LIGHT UI
    # ------------------------------------------------------------

    with gr.Column(elem_classes=["paper-section"]):

        gr.Markdown(
            "### Consult the AI Advisory Assistant",
            elem_classes=["section-title"]
        )

        gr.Markdown(
            "Ask questions about the currently loaded company and validated metrics.",
            elem_classes=["section-subtitle"]
        )

        chat_state = gr.State([])

        chat_display = gr.HTML(
            render_chat_html([]),
            elem_classes=["chat-shell"]
        )

        with gr.Row(elem_classes=["chat-input-row"]):

            chat_input = gr.Textbox(
                placeholder=(
                    "Ask about liquidity, leverage, distress risk, "
                    "stress testing..."
                ),
                show_label=False,
                lines=2,
                scale=7
            )

            chat_send = gr.Button(
                "Send",
                elem_classes=["chat-send"],
                scale=1
            )

            chat_clear = gr.Button(
                "Clear",
                elem_classes=["chat-clear"],
                scale=1
            )

        chat_send.click(
            fn=submit_chat,
            inputs=[
                chat_input,
                chat_state,
                interest_shock,
            ],
            outputs=[
                chat_input,
                chat_state,
                chat_display,
            ]
        )

        chat_input.submit(
            fn=submit_chat,
            inputs=[
                chat_input,
                chat_state,
                interest_shock,
            ],
            outputs=[
                chat_input,
                chat_state,
                chat_display,
            ]
        )

        chat_clear.click(
            fn=clear_chat,
            inputs=[],
            outputs=[
                chat_state,
                chat_display,
            ]
        )

    # ------------------------------------------------------------
    # DISCLAIMER
    # ------------------------------------------------------------

    gr.HTML("""
    <div class="disclaimer">
        <b>Important:</b> This dashboard is a screening and analytics
        framework, not a formal credit rating or investment recommendation.
        Screener and Yahoo Finance are used as secondary convenience sources.
        For final academic submission, independently verify material figures
        against the company's audited annual report and official NSE/BSE filings.
    </div>
    """)


    # ============================================================
    # EVENTS
    # ============================================================

    fetch_outputs = [
        status_box,
        source_box,
        audit_table,
        metrics_grid,
    ]

    fetch_btn.click(
        fn=fetch_company_data,
        inputs=[
            nifty_dropdown,
            custom_input,
        ],
        outputs=fetch_outputs
    )


    analyze_btn.click(
        fn=lambda shock: (
            build_analysis_report(
                CURRENT_DATASET,
                shock
            )[0]
            if CURRENT_DATASET.get("symbol")
            else "⚠️ Extract a company first."
        ),
        inputs=[interest_shock],
        outputs=[report_output]
    )


    ai_btn.click(
        fn=generate_ai_insight,
        inputs=[interest_shock],
        outputs=[ai_output]
    )


# ================================================================
# 16. LAUNCH
# ================================================================

print("=" * 80)
print("INSTITUTIONAL DISTRESS INTELLIGENCE ENGINE")
print("=" * 80)
print("✅ Two-source validation layer enabled")
print("✅ Missing values are NOT replaced with fake defaults")
print("✅ Cross-source discrepancies are flagged")
print("✅ Full Beneish model used only when two-year inputs exist")
print("✅ Altman is restricted for financial institutions")
print("✅ Gemini is interpretation-only")
print("✅ Gemini uses current text models with retry + fallback")
print("✅ Audit table uses custom HTML — no Gradio Dataframe theme leakage")
print("✅ Chat assistant uses custom HTML — no Gradio ChatInterface theme leakage")
print("✅ Gradio theme pinned + light-mode CSS lock enabled")
print("=" * 80)

try:
    demo.launch(
        share=True,
        debug=True,
        prevent_thread_lock=True
    )
except Exception as launch_error:
    print(f"⚠️ Initial Gradio launch failed: {launch_error}")
    print("Trying one clean launch without debug mode...")
    demo.launch(
        share=True,
        debug=False,
        prevent_thread_lock=True
    )
