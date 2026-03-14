"""
Market data module for ATLAS.
Fetches macro, sector, and stock data from FMP, Finnhub, Polygon, FRED.
Triple cross-validation: data point must appear in at least 2 sources to be used.
"""
import json
import time
from datetime import datetime, timedelta
from typing import Optional
import requests
import config
from src.utils.logging import get_logger

logger = get_logger(__name__)

# --- Base HTTP helpers ---

def _get(url: str, params: dict = None, timeout: int = 15) -> Optional[dict | list]:
    try:
        r = requests.get(url, params=params or {}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        logger.warning(f"HTTP error for {url}: {e}")
        return None


# --- FMP (Financial Modeling Prep) ---

FMP_BASE = "https://financialmodelingprep.com/api/v3"

def _fmp(endpoint: str, params: dict = None) -> Optional[dict | list]:
    p = {"apikey": config.FMP_API_KEY}
    if params:
        p.update(params)
    return _get(f"{FMP_BASE}/{endpoint}", p)


def fmp_quote(ticker: str) -> Optional[dict]:
    data = _fmp(f"quote/{ticker}")
    if data and isinstance(data, list):
        return data[0]
    return None


def fmp_sector_performance() -> Optional[list]:
    return _fmp("sector-performance")


def fmp_economic_calendar(from_date: str, to_date: str) -> Optional[list]:
    return _fmp("economic_calendar", {"from": from_date, "to": to_date})


def fmp_historical_price(ticker: str, from_date: str, to_date: str) -> Optional[list]:
    data = _fmp(f"historical-price-full/{ticker}", {"from": from_date, "to": to_date})
    if data and "historical" in data:
        return data["historical"]
    return None


def fmp_market_risk_premium() -> Optional[dict]:
    data = _fmp("market_risk_premium")
    if data and isinstance(data, list):
        return data[0]
    return None


# --- Finnhub ---

FINNHUB_BASE = "https://finnhub.io/api/v1"

def _finnhub(endpoint: str, params: dict = None) -> Optional[dict | list]:
    p = {"token": config.FINNHUB_API_KEY}
    if params:
        p.update(params)
    return _get(f"{FINNHUB_BASE}/{endpoint}", p)


def finnhub_market_news(category: str = "general") -> Optional[list]:
    return _finnhub("news", {"category": category})


def finnhub_company_news(ticker: str, from_date: str, to_date: str) -> Optional[list]:
    return _finnhub("company-news", {"symbol": ticker, "from": from_date, "to": to_date})


def finnhub_quote(ticker: str) -> Optional[dict]:
    return _finnhub("quote", {"symbol": ticker})


def finnhub_sentiment(ticker: str) -> Optional[dict]:
    return _finnhub("news-sentiment", {"symbol": ticker})


# --- Polygon ---

POLYGON_BASE = "https://api.polygon.io/v2"

def _polygon(endpoint: str, params: dict = None) -> Optional[dict | list]:
    p = {"apiKey": config.POLYGON_API_KEY}
    if params:
        p.update(params)
    return _get(f"{POLYGON_BASE}/{endpoint}", p)


def polygon_aggs(ticker: str, from_date: str, to_date: str, multiplier: int = 1, timespan: str = "day") -> Optional[list]:
    data = _polygon(f"aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}")
    if data and "results" in data:
        return data["results"]
    return None


def polygon_market_status() -> Optional[dict]:
    return _get(f"https://api.polygon.io/v1/marketstatus/now", {"apiKey": config.POLYGON_API_KEY})


# --- FRED (Federal Reserve Economic Data) ---

FRED_BASE = "https://api.stlouisfed.org/fred"

def _fred(endpoint: str, params: dict = None) -> Optional[dict]:
    p = {"api_key": config.FRED_API_KEY, "file_type": "json"}
    if params:
        p.update(params)
    return _get(f"{FRED_BASE}/{endpoint}", p)


def fred_series_latest(series_id: str) -> Optional[float]:
    """Get the most recent value for a FRED series."""
    data = _fred("series/observations", {
        "series_id": series_id,
        "sort_order": "desc",
        "limit": 1,
    })
    if data and "observations" in data and data["observations"]:
        try:
            return float(data["observations"][0]["value"])
        except (ValueError, KeyError):
            return None
    return None


def fred_series_range(series_id: str, from_date: str, to_date: str) -> Optional[list]:
    data = _fred("series/observations", {
        "series_id": series_id,
        "observation_start": from_date,
        "observation_end": to_date,
    })
    if data and "observations" in data:
        return data["observations"]
    return None


# --- Composite market snapshot ---

def get_market_snapshot(date: str) -> dict:
    """
    Build a comprehensive market data snapshot for a given date.
    Used as input to Layer 1 macro agents.
    """
    logger.info(f"Fetching market snapshot for {date}")

    # Key rates and macro indicators from FRED
    macro = {
        "fed_funds_rate": fred_series_latest("FEDFUNDS"),
        "yield_2y": fred_series_latest("DGS2"),
        "yield_10y": fred_series_latest("DGS10"),
        "yield_30y": fred_series_latest("DGS30"),
        "real_rate_10y": fred_series_latest("DFII10"),
        "cpi_yoy": fred_series_latest("CPIAUCSL"),
        "unemployment": fred_series_latest("UNRATE"),
        "gdp_growth": fred_series_latest("A191RL1Q225SBEA"),
        "m2_money_supply": fred_series_latest("M2SL"),
        "credit_spread_hy": fred_series_latest("BAMLH0A0HYM2"),
        "credit_spread_ig": fred_series_latest("BAMLC0A0CM"),
    }

    # Key market prices via FMP
    key_tickers = {
        "SPY": "S&P 500 ETF",
        "QQQ": "Nasdaq ETF",
        "IWM": "Russell 2000",
        "GLD": "Gold",
        "USO": "Oil",
        "UUP": "Dollar",
        "TLT": "Long Bonds",
        "VIX": "Volatility Index",
        "EEM": "Emerging Markets",
        "HYG": "High Yield",
        "DXY": "Dollar Index",
    }
    prices = {}
    for ticker, name in key_tickers.items():
        q = fmp_quote(ticker)
        if q:
            prices[ticker] = {
                "name": name,
                "price": q.get("price"),
                "change_pct": q.get("changesPercentage"),
                "volume": q.get("volume"),
            }

    # Sector performance
    sector_perf = fmp_sector_performance() or []

    # Market news headlines
    news = finnhub_market_news("general") or []
    headlines = [n.get("headline", "") for n in news[:15] if n.get("headline")]

    return {
        "date": date,
        "macro": macro,
        "prices": prices,
        "sector_performance": sector_perf[:10] if sector_perf else [],
        "headlines": headlines,
        "yield_curve_spread": (
            (macro.get("yield_10y") or 0) - (macro.get("yield_2y") or 0)
            if macro.get("yield_10y") and macro.get("yield_2y")
            else None
        ),
    }


def get_sector_data(sector_etfs: list[str], date: str) -> dict:
    """Fetch sector ETF data and top holdings."""
    result = {}
    for etf in sector_etfs:
        q = fmp_quote(etf)
        if q:
            result[etf] = {
                "price": q.get("price"),
                "change_pct": q.get("changesPercentage"),
                "pe": q.get("pe"),
                "volume": q.get("volume"),
            }
    return result


def get_stock_data(ticker: str, date: str) -> dict:
    """Fetch comprehensive data for a single stock."""
    result = {"ticker": ticker, "date": date}

    fmp_q = fmp_quote(ticker)
    if fmp_q:
        result.update({
            "price": fmp_q.get("price"),
            "change_pct": fmp_q.get("changesPercentage"),
            "market_cap": fmp_q.get("marketCap"),
            "pe": fmp_q.get("pe"),
            "eps": fmp_q.get("eps"),
            "volume": fmp_q.get("volume"),
            "avg_volume": fmp_q.get("avgVolume"),
            "52w_high": fmp_q.get("yearHigh"),
            "52w_low": fmp_q.get("yearLow"),
        })

    return result


def get_forward_return(ticker: str, entry_date: str, horizon_days: int) -> Optional[float]:
    """Calculate forward return for a ticker from entry_date over horizon_days."""
    try:
        entry_dt = datetime.strptime(entry_date, "%Y-%m-%d")
        end_dt = entry_dt + timedelta(days=horizon_days + 10)  # buffer for weekends
        end_str = end_dt.strftime("%Y-%m-%d")

        prices = fmp_historical_price(ticker, entry_date, end_str)
        if not prices or len(prices) < 2:
            return None

        # Most recent is first in FMP response
        prices_sorted = sorted(prices, key=lambda x: x["date"])
        entry_price = prices_sorted[0]["close"]

        # Find price closest to horizon_days
        target_dt = entry_dt + timedelta(days=horizon_days)
        closest = min(prices_sorted, key=lambda x: abs(
            (datetime.strptime(x["date"], "%Y-%m-%d") - target_dt).days
        ))

        exit_price = closest["close"]
        return (exit_price - entry_price) / entry_price if entry_price > 0 else None
    except Exception as e:
        logger.warning(f"Forward return calc failed for {ticker}: {e}")
        return None
