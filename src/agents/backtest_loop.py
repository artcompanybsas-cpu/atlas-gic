"""
Main backtest loop: runs the full ATLAS system across historical dates.
One iteration = one trading day.
"""
import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import config
from src.agents.market_data import get_market_snapshot, get_stock_data
from src.agents.eod_cycle import run_full_cycle
from src.agents.scorecard import update_forward_returns, update_darwinian_weights
from src.agents.autoresearch import maybe_run_autoresearch
from src.utils.logging import get_logger

logger = get_logger(__name__)

PORTFOLIO_FILE = config.DATA_DIR / "state" / "portfolio.json"
TRAJECTORY_FILE = config.DATA_DIR / "backtest" / "portfolio_trajectory.csv"


# --- Portfolio management ---

def _load_portfolio() -> dict:
    if PORTFOLIO_FILE.exists():
        with open(PORTFOLIO_FILE) as f:
            return json.load(f)
    return {
        "cash": config.STARTING_CAPITAL,
        "positions": {},  # ticker -> {shares, avg_price, entry_date}
        "total_value": config.STARTING_CAPITAL,
        "trade_history": [],
    }


def _save_portfolio(portfolio: dict) -> None:
    PORTFOLIO_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=2)


def _mark_portfolio_to_market(portfolio: dict, date: str) -> dict:
    """Update portfolio total value based on current prices."""
    total = portfolio["cash"]
    for ticker, pos in portfolio["positions"].items():
        data = get_stock_data(ticker, date)
        price = data.get("price") or pos.get("avg_price", 0)
        total += pos["shares"] * price
    portfolio["total_value"] = total
    return portfolio


def _execute_actions(portfolio: dict, actions: list, date: str) -> dict:
    """Execute portfolio actions from the CIO."""
    for action in actions:
        ticker = action.get("ticker")
        act = action.get("action", "").upper()
        shares = int(action.get("shares", 0))

        if not ticker or not act or shares <= 0:
            continue

        data = get_stock_data(ticker, date)
        price = data.get("price")

        if not price:
            logger.warning(f"No price data for {ticker}, skipping {act}")
            continue

        if act == "BUY":
            cost = shares * price
            if cost > portfolio["cash"]:
                # Reduce size to fit available cash
                shares = int(portfolio["cash"] * 0.95 / price)
                cost = shares * price
            if shares <= 0:
                continue

            existing = portfolio["positions"].get(ticker, {"shares": 0, "avg_price": 0, "entry_date": date})
            total_shares = existing["shares"] + shares
            if total_shares > 0:
                avg_price = (existing["shares"] * existing["avg_price"] + shares * price) / total_shares
            else:
                avg_price = price

            portfolio["positions"][ticker] = {
                "shares": total_shares,
                "avg_price": avg_price,
                "entry_date": existing.get("entry_date", date),
            }
            portfolio["cash"] -= cost
            logger.info(f"BUY {shares} {ticker} @ ${price:.2f} (cost: ${cost:,.0f})")

        elif act == "SELL":
            if ticker not in portfolio["positions"]:
                continue
            pos = portfolio["positions"][ticker]
            sell_shares = min(shares, pos["shares"])
            proceeds = sell_shares * price
            portfolio["cash"] += proceeds
            pos["shares"] -= sell_shares

            if pos["shares"] <= 0:
                del portfolio["positions"][ticker]

            logger.info(f"SELL {sell_shares} {ticker} @ ${price:.2f} (proceeds: ${proceeds:,.0f})")

        portfolio["trade_history"].append({
            "date": date,
            "ticker": ticker,
            "action": act,
            "shares": shares,
            "price": price,
        })

    return portfolio


def _append_trajectory(day: int, date: str, portfolio: dict, prev_value: float) -> None:
    """Append one row to the trajectory CSV."""
    TRAJECTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    value = portfolio["total_value"]
    daily_return = (value / prev_value - 1) * 100 if prev_value > 0 else 0
    cumulative_return = (value / config.STARTING_CAPITAL - 1) * 100

    file_exists = TRAJECTORY_FILE.exists()
    with open(TRAJECTORY_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["day", "date", "portfolio_value", "daily_return_pct", "cumulative_return_pct"])
        writer.writerow([day, date, f"{value:.2f}", f"{daily_return:.4f}", f"{cumulative_return:.4f}"])


def _get_trading_dates(start_date: str, end_date: str) -> list[str]:
    """Generate weekday dates between start and end (approximate - no holiday calendar)."""
    dates = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    while current <= end:
        if current.weekday() < 5:  # Mon-Fri
            dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


# --- Main backtest loop ---

def run_backtest(start_date: str, end_date: str, resume: bool = False) -> dict:
    """
    Run the full backtest from start_date to end_date.
    If resume=True, continue from last portfolio state.
    """
    dates = _get_trading_dates(start_date, end_date)
    logger.info(f"Starting backtest: {start_date} to {end_date} ({len(dates)} trading days)")

    portfolio = _load_portfolio() if resume else {
        "cash": config.STARTING_CAPITAL,
        "positions": {},
        "total_value": config.STARTING_CAPITAL,
        "trade_history": [],
    }

    if not resume:
        _save_portfolio(portfolio)
        # Clear old trajectory
        if TRAJECTORY_FILE.exists():
            TRAJECTORY_FILE.unlink()

    prev_value = portfolio["total_value"]

    for day_num, date in enumerate(dates, start=1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Day {day_num}/{len(dates)}: {date}")
        logger.info(f"Portfolio value: ${portfolio['total_value']:,.0f}")

        try:
            # 1. Update forward returns for past recommendations
            update_forward_returns(date)

            # 2. Fetch market data
            market_data = get_market_snapshot(date)

            # 3. Run 4-layer agent cycle
            cio_decisions = run_full_cycle(market_data, portfolio)

            # 4. Execute trades
            actions = cio_decisions.get("actions") or cio_decisions.get("portfolio_actions", [])
            portfolio = _execute_actions(portfolio, actions, date)

            # 5. Mark to market
            portfolio = _mark_portfolio_to_market(portfolio, date)

            # 6. Update Darwinian weights
            update_darwinian_weights(date)

            # 7. Maybe run autoresearch
            maybe_run_autoresearch(date, day_num)

            # 8. Save state
            _save_portfolio(portfolio)
            _append_trajectory(day_num, date, portfolio, prev_value)
            prev_value = portfolio["total_value"]

        except KeyboardInterrupt:
            logger.info("Backtest interrupted by user")
            break
        except Exception as e:
            logger.error(f"Error on day {day_num} ({date}): {e}", exc_info=True)
            # Continue to next day rather than crashing
            _append_trajectory(day_num, date, portfolio, prev_value)
            continue

    final_return = (portfolio["total_value"] / config.STARTING_CAPITAL - 1) * 100
    logger.info(f"\n{'='*60}")
    logger.info(f"Backtest complete!")
    logger.info(f"Days run: {day_num}")
    logger.info(f"Final value: ${portfolio['total_value']:,.0f}")
    logger.info(f"Total return: {final_return:.1f}%")

    return {
        "start_date": start_date,
        "end_date": dates[day_num - 1] if dates else end_date,
        "trading_days": day_num,
        "starting_value": config.STARTING_CAPITAL,
        "ending_value": portfolio["total_value"],
        "total_return_pct": final_return,
    }
