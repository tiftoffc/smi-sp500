#!/usr/bin/env python3
"""
SMI Scanner for S&P 500
Filter: SMI(10,3,3) <= -40 and SMI > Signal
Additionally calculates RSL (26T / 26W) and RSL vs Sektor.
Outputs JSON for the WordPress shortcode.
"""

import json
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def get_sp500_tickers():
    """Fetch S&P 500 tickers from a stable GitHub source."""
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    df = pd.read_csv(url)
    df['yf_ticker'] = df['Symbol'].str.replace('.', '-', regex=False)
    if 'Sector' in df.columns:
        df['GICS Sector'] = df['Sector']
    if 'Name' in df.columns:
        df['Security'] = df['Name']
    print(f"Loaded {len(df)} S&P 500 tickers from stable source")
    return df

def calculate_smi(high, low, close, period=10, smooth1=3, smooth2=3, signal_period=3):
    """Calculate Stochastic Momentum Index (SMI) and signal line."""
    hh = high.rolling(window=period, min_periods=period).max()
    ll = low.rolling(window=period, min_periods=period).min()
    midpoint = (hh + ll) / 2.0
    distance = close - midpoint
    rng = hh - ll

    dist1 = distance.ewm(span=smooth1, adjust=False, min_periods=smooth1).mean()
    dist2 = dist1.ewm(span=smooth2, adjust=False, min_periods=smooth2).mean()
    rng1 = rng.ewm(span=smooth1, adjust=False, min_periods=smooth1).mean()
    rng2 = rng1.ewm(span=smooth2, adjust=False, min_periods=smooth2).mean()

    smi = 100.0 * dist2 / (0.5 * rng2.replace(0, pd.NA))
    signal = smi.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
    return smi, signal

def get_fundamentals(ticker_obj):
    """Extract market cap, revenue, net income from yfinance info."""
    try:
        info = ticker_obj.info
        return {
            'name': info.get('shortName') or info.get('longName') or '',
            'sector': info.get('sector') or '',
            'industry': info.get('industry') or '',
            'market_cap': info.get('marketCap'),
            'revenue': info.get('totalRevenue'),
            'net_income': info.get('netIncomeToCommon') or info.get('netIncome'),
        }
    except Exception:
        return {
            'name': '', 'sector': '', 'industry': '',
            'market_cap': None, 'revenue': None, 'net_income': None
        }

def scan():
    print("Fetching S&P 500 list...")
    sp500 = get_sp500_tickers()
    results = []

    for idx, row in sp500.iterrows():
        symbol = row['Symbol']
        yf_ticker = row['yf_ticker']
        print(f"[{idx+1}/{len(sp500)}] {symbol}", end='\r')

        try:
            ticker = yf.Ticker(yf_ticker)
            hist = ticker.history(period="1y", interval="1d", auto_adjust=True)
            if hist.empty or len(hist) < 40:
                continue

            high = hist['High']
            low = hist['Low']
            close = hist['Close']

            smi, signal = calculate_smi(high, low, close)
            if smi.empty or pd.isna(smi.iloc[-1]) or pd.isna(signal.iloc[-1]):
                continue

            last_smi = float(smi.iloc[-1])
            last_signal = float(signal.iloc[-1])
            diff = last_smi - last_signal

            if not (last_smi <= -40 and last_smi > last_signal):
                continue

            pct_1d = None
            pct_1w = None
            if len(close) >= 2:
                pct_1d = float((close.iloc[-1] / close.iloc[-2] - 1) * 100)
            if len(close) >= 6:
                pct_1w = float((close.iloc[-1] / close.iloc[-6] - 1) * 100)

            rsl_26d = None
            rsl_26w = None
            if len(close) >= 27:
                rsl_26d = float((close.iloc[-1] / close.iloc[-27] - 1) * 100)
            if len(close) >= 131:
                rsl_26w = float((close.iloc[-1] / close.iloc[-131] - 1) * 100)

            fund = get_fundamentals(ticker)

            results.append({
                'ticker': symbol,
                'name': fund['name'],
                'sector': fund['sector'] or row.get('GICS Sector', ''),
                'industry': fund['industry'] or row.get('GICS Sub-Industry', ''),
                'smi': round(last_smi, 2),
                'signal': round(last_signal, 2),
                'diff': round(diff, 2),
                'market_cap': fund['market_cap'],
                'revenue': fund['revenue'],
                'net_income': fund['net_income'],
                'pct_1d': round(pct_1d, 2) if pct_1d is not None else None,
