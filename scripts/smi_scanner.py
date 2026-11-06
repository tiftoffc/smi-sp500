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
import requests
warnings.filterwarnings('ignore')

def get_sp500_tickers():
    """Fetch current S&P 500 tickers + basic info from Wikipedia."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    tables = pd.read_html(response.text)
    df = tables[0]
    # Clean tickers (BRK.B -> BRK-B for yfinance)
    df['yf_ticker'] = df['Symbol'].str.replace('.', '-', regex=False)
    return df

def calculate_smi(high, low, close, period=10, smooth1=3, smooth2=3, signal_period=3):
    """Calculate Stochastic Momentum Index (SMI) and signal line."""
    hh = high.rolling(window=period, min_periods=period).max()
    ll = low.rolling(window=period, min_periods=period).min()
    midpoint = (hh + ll) / 2.0
    distance = close - midpoint
    rng = hh - ll

    # Double EMA smoothing
    dist1 = distance.ewm(span=smooth1, adjust=False, min_periods=smooth1).mean()
    dist2 = dist1.ewm(span=smooth2, adjust=False, min_periods=smooth2).mean()
    rng1 = rng.ewm(span=smooth1, adjust=False, min_periods=smooth1).mean()
    rng2 = rng1.ewm(span=smooth2, adjust=False, min_periods=smooth2).mean()

    # Avoid division by zero
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
            # Need enough history for SMI warm-up + 26 weeks
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

            # Main filter: oversold + short-term > long-term
            if not (last_smi <= -40 and last_smi > last_signal):
                continue

            # Performance
            pct_1d = None
            pct_1w = None
            if len(close) >= 2:
                pct_1d = float((close.iloc[-1] / close.iloc[-2] - 1) * 100)
            if len(close) >= 6:
                pct_1w = float((close.iloc[-1] / close.iloc[-6] - 1) * 100)

            # RSL: 26-day and ~26-week performance
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
                'pct_1w': round(pct_1w, 2) if pct_1w is not None else None,
                'rsl_26d': round(rsl_26d, 2) if rsl_26d is not None else None,
                'rsl_26w': round(rsl_26w, 2) if rsl_26w is not None else None,
                'rsl_vs_sector_26d': None,
                'rsl_vs_sector_26w': None,
            })

        except Exception:
            continue

    print(f"\nFound {len(results)} matches. Calculating RSL vs Sektor...")

    # Aggregate sector averages for RSL vs Sektor
    sector_perf = defaultdict(lambda: {'26d': [], '26w': []})
    for s in results:
        sec = s.get('sector')
        if not sec:
            continue
        if s['rsl_26d'] is not None:
            sector_perf[sec]['26d'].append(s['rsl_26d'])
        if s['rsl_26w'] is not None:
            sector_perf[sec]['26w'].append(s['rsl_26w'])

    sector_means = {}
    for sec, vals in sector_perf.items():
        sector_means[sec] = {
            '26d': sum(vals['26d']) / len(vals['26d']) if vals['26d'] else None,
            '26w': sum(vals['26w']) / len(vals['26w']) if vals['26w'] else None,
        }

    for s in results:
        sec = s.get('sector')
        if sec and sec in sector_means:
            if s['rsl_26d'] is not None and sector_means[sec]['26d'] is not None:
                s['rsl_vs_sector_26d'] = round(s['rsl_26d'] - sector_means[sec]['26d'], 2)
            if s['rsl_26w'] is not None and sector_means[sec]['26w'] is not None:
                s['rsl_vs_sector_26w'] = round(s['rsl_26w'] - sector_means[sec]['26w'], 2)

    output = {
        'updated': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
        'stocks': sorted(results, key=lambda x: x['smi'])  # most oversold first
    }

    with open('public/smi_scan.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved public/smi_scan.json with {len(results)} titles")
    return output

if __name__ == '__main__':
    scan()
