"""
Agent scorecard: tracks every recommendation, calculates rolling Sharpe ratios,
and updates Darwinian weights after each trading day.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
import numpy as np
import config
from src.utils.logging import get_logger

logger = get_logger(__name__)

SCORECARD_FILE = config.DATA_DIR / "track_record" / "scorecard.json"
WEIGHTS_FILE = config.DATA_DIR / "state" / "darwinian_weights.json"


# --- Data structures ---

def _empty_scorecard() -> dict:
    return {
        "recommendations": [],  # List of recommendation records
        "agent_stats": {agent: {"total": 0, "scored": 0} for agent in config.ALL_AGENTS},
    }


def load_scorecard() -> dict:
    if SCORECARD_FILE.exists():
        with open(SCORECARD_FILE) as f:
            return json.load(f)
    return _empty_scorecard()


def save_scorecard(sc: dict) -> None:
    SCORECARD_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SCORECARD_FILE, "w") as f:
        json.dump(sc, f, indent=2)


def load_darwinian_weights() -> dict:
    if WEIGHTS_FILE.exists():
        with open(WEIGHTS_FILE) as f:
            return json.load(f)
    # Start with defaults from backtest
    weights = dict(config.DEFAULT_DARWINIAN_WEIGHTS)
    save_darwinian_weights(weights)
    return weights


def save_darwinian_weights(weights: dict) -> None:
    WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(WEIGHTS_FILE, "w") as f:
        json.dump(weights, f, indent=2)


# --- Recommendation logging ---

def log_recommendation(
    agent_name: str,
    date: str,
    ticker: str,
    direction: str,  # "LONG" or "SHORT"
    conviction: int,  # 1-100
    entry_price: Optional[float] = None,
) -> None:
    """Record an agent's recommendation for later scoring."""
    sc = load_scorecard()
    rec = {
        "agent": agent_name,
        "date": date,
        "ticker": ticker,
        "direction": direction,
        "conviction": conviction,
        "entry_price": entry_price,
        "return_1d": None,
        "return_5d": None,
        "return_20d": None,
        "scored": False,
    }
    sc["recommendations"].append(rec)
    sc["agent_stats"][agent_name]["total"] += 1
    save_scorecard(sc)


def update_forward_returns(date: str) -> None:
    """
    Update forward returns for recommendations where enough time has elapsed.
    Called at the start of each backtest day.
    """
    from src.agents.market_data import get_forward_return

    sc = load_scorecard()
    today = datetime.strptime(date, "%Y-%m-%d")
    updated = 0

    for rec in sc["recommendations"]:
        if rec["scored"]:
            continue

        rec_date = datetime.strptime(rec["date"], "%Y-%m-%d")
        elapsed_days = (today - rec_date).days

        # Update returns as time horizons become available
        if elapsed_days >= 1 and rec["return_1d"] is None:
            rec["return_1d"] = get_forward_return(rec["ticker"], rec["date"], 1)

        if elapsed_days >= 7 and rec["return_5d"] is None:
            rec["return_5d"] = get_forward_return(rec["ticker"], rec["date"], 5)

        if elapsed_days >= 30 and rec["return_20d"] is None:
            rec["return_20d"] = get_forward_return(rec["ticker"], rec["date"], 20)
            # Mark as fully scored once 20d return is available
            if rec["return_20d"] is not None:
                rec["scored"] = True
                sc["agent_stats"][rec["agent"]]["scored"] += 1
                updated += 1

    if updated:
        save_scorecard(sc)
        logger.info(f"Updated forward returns for {updated} recommendations")


def calculate_agent_sharpe(agent_name: str, lookback_days: int = 60, horizon: str = "5d") -> Optional[float]:
    """
    Calculate rolling Sharpe ratio for an agent over the lookback window.
    horizon: "1d", "5d", or "20d"
    """
    sc = load_scorecard()
    field = f"return_{horizon}"

    # Get scored recommendations within lookback
    cutoff = datetime.now()
    relevant = [
        r for r in sc["recommendations"]
        if r["agent"] == agent_name
        and r.get(field) is not None
    ]

    if len(relevant) < 5:
        return None

    # Weight by conviction and flip sign for shorts
    returns = []
    for rec in relevant[-lookback_days:]:
        raw_return = rec[field]
        weighted = raw_return * (rec["conviction"] / 100)
        if rec["direction"] == "SHORT":
            weighted *= -1
        returns.append(weighted)

    if len(returns) < 3:
        return None

    arr = np.array(returns)
    std = np.std(arr)
    if std == 0:
        return 0.0
    return float(np.mean(arr) / std)


def get_all_agent_sharpes(lookback_days: int = 60) -> dict:
    """Return Sharpe ratios for all agents."""
    return {
        agent: calculate_agent_sharpe(agent, lookback_days)
        for agent in config.ALL_AGENTS
    }


def get_worst_agent(lookback_days: int = 60, exclude_recently_modified: set = None) -> Optional[str]:
    """Return the agent with the lowest Sharpe ratio (candidate for autoresearch)."""
    sharpes = get_all_agent_sharpes(lookback_days)
    exclude = exclude_recently_modified or set()

    # Filter to agents with enough data
    candidates = {
        agent: s for agent, s in sharpes.items()
        if s is not None and agent not in exclude
    }

    if not candidates:
        return None

    return min(candidates, key=lambda a: candidates[a])


# --- Darwinian weight updates ---

def update_darwinian_weights(date: str) -> dict:
    """
    After each trading day, update agent weights:
    - Top quartile performers: weight * 1.05 (capped at 2.5)
    - Bottom quartile performers: weight * 0.95 (floored at 0.3)
    """
    sharpes = get_all_agent_sharpes(lookback_days=20)

    # Only update agents with enough data
    scored_agents = {a: s for a, s in sharpes.items() if s is not None}

    if len(scored_agents) < 4:
        logger.info("Not enough scored agents for Darwinian weight update")
        return load_darwinian_weights()

    weights = load_darwinian_weights()
    sorted_agents = sorted(scored_agents.items(), key=lambda x: x[1])
    n = len(sorted_agents)
    quartile = max(1, n // 4)

    bottom_agents = {a for a, _ in sorted_agents[:quartile]}
    top_agents = {a for a, _ in sorted_agents[-quartile:]}

    for agent in scored_agents:
        if agent in top_agents:
            weights[agent] = min(
                config.DARWINIAN_WEIGHT_CEILING,
                weights.get(agent, 1.0) * config.DARWINIAN_REWARD_MULTIPLIER,
            )
        elif agent in bottom_agents:
            weights[agent] = max(
                config.DARWINIAN_WEIGHT_FLOOR,
                weights.get(agent, 1.0) * config.DARWINIAN_PENALTY_MULTIPLIER,
            )

    save_darwinian_weights(weights)
    logger.info(f"Updated Darwinian weights on {date}")

    # Log top and bottom
    top_str = ", ".join(f"{a}={weights[a]:.2f}" for a in top_agents)
    bot_str = ", ".join(f"{a}={weights[a]:.2f}" for a in bottom_agents)
    logger.info(f"Top quartile: {top_str}")
    logger.info(f"Bottom quartile: {bot_str}")

    return weights
