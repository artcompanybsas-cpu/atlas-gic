"""
End-of-day cycle: orchestrates all 25 agents through 4 layers.
Each layer feeds structured JSON to the next.
"""
import json
import asyncio
from pathlib import Path
from typing import Optional
import config
from src.utils.llm import call_agent
from src.utils.logging import get_logger, log_agent_output
from src.agents.scorecard import log_recommendation, load_darwinian_weights

logger = get_logger(__name__)

STATE_DIR = config.DATA_DIR / "state"


def _load_prompt(agent_name: str) -> str:
    """Load an agent's prompt file from the prompts directory."""
    for layer_dir in ["layer1", "layer2", "layer3", "layer4"]:
        path = config.PROMPTS_DIR / layer_dir / f"{agent_name}.md"
        if path.exists():
            return path.read_text()
    raise FileNotFoundError(f"Prompt not found for agent: {agent_name}")


def _state_path(filename: str) -> Path:
    return STATE_DIR / filename


def _save_state(filename: str, data: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_state_path(filename), "w") as f:
        json.dump(data, f, indent=2)


def _load_state(filename: str) -> dict:
    path = _state_path(filename)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# --- Layer 1: Macro agents ---

def run_macro_agent(agent_name: str, market_data: dict, darwinian_weights: dict) -> dict:
    """Run a single Layer 1 macro agent."""
    prompt = _load_prompt(agent_name)
    weight = darwinian_weights.get(agent_name, 1.0)

    user_msg = f"""Date: {market_data.get('date', 'unknown')}
Darwinian weight: {weight:.2f}

Market Data:
{json.dumps(market_data, indent=2)}

Analyse the data and provide your regime signal. Return ONLY valid JSON.
"""
    result = call_agent(prompt, user_msg, agent_name=agent_name)
    log_agent_output(agent_name, market_data.get("date", ""), result)
    return result


def run_layer1(market_data: dict, darwinian_weights: dict) -> dict:
    """
    Run all 10 macro agents and aggregate into a single regime signal.
    Returns: {"regime": "RISK_ON|RISK_OFF|NEUTRAL", "signals": {...}, "weighted_conviction": 0-100}
    """
    logger.info("=== Layer 1: Macro agents ===")
    signals = {}

    for agent in config.LAYER1_AGENTS:
        logger.info(f"Running {agent}")
        result = run_macro_agent(agent, market_data, darwinian_weights)
        signals[agent] = result

    # Aggregate regime using Darwinian weights
    regime_votes = {"RISK_ON": 0.0, "RISK_OFF": 0.0, "NEUTRAL": 0.0}
    total_weight = 0.0

    for agent, signal in signals.items():
        regime = signal.get("regime") or signal.get("signal", "NEUTRAL")
        if regime not in regime_votes:
            regime = "NEUTRAL"
        conviction = signal.get("conviction", 50)
        weight = darwinian_weights.get(agent, 1.0)
        regime_votes[regime] += weight * conviction
        total_weight += weight

    # Determine consensus regime
    if total_weight > 0:
        winning_regime = max(regime_votes, key=lambda r: regime_votes[r])
        weighted_conviction = int(regime_votes[winning_regime] / total_weight)
    else:
        winning_regime = "NEUTRAL"
        weighted_conviction = 50

    macro_output = {
        "regime": winning_regime,
        "signals": signals,
        "regime_votes": regime_votes,
        "weighted_conviction": weighted_conviction,
        "date": market_data.get("date"),
    }

    _save_state("macro_regime.json", macro_output)
    logger.info(f"Macro regime: {winning_regime} (conviction: {weighted_conviction})")
    return macro_output


# --- Layer 2: Sector desks ---

SECTOR_ETFS = {
    "semiconductor": ["SOXX", "SMH", "NVDA", "AMD", "AVGO", "TSM", "AMAT"],
    "energy": ["XLE", "XOM", "CVX", "SLB", "OXY", "COP"],
    "biotech": ["XBI", "IBB", "AMGN", "GILD", "MRNA", "REGN"],
    "consumer": ["XLY", "XLP", "AMZN", "TSLA", "COST", "WMT", "HD"],
    "industrials": ["XLI", "RTX", "LMT", "GE", "CAT", "UNP"],
    "financials": ["XLF", "JPM", "BAC", "GS", "MS", "BRK-B"],
    "relationship_mapper": [],
}


def run_sector_agent(agent_name: str, macro_regime: dict, market_data: dict, darwinian_weights: dict) -> dict:
    """Run a single Layer 2 sector agent."""
    prompt = _load_prompt(agent_name)
    weight = darwinian_weights.get(agent_name, 1.0)

    user_msg = f"""Date: {market_data.get('date', 'unknown')}
Darwinian weight: {weight:.2f}

Macro Regime from Layer 1:
{json.dumps(macro_regime, indent=2)}

Market Data:
{json.dumps(market_data, indent=2)}

Analyse your sector and provide long/short recommendations. Return ONLY valid JSON.
"""
    result = call_agent(prompt, user_msg, agent_name=agent_name)
    log_agent_output(agent_name, market_data.get("date", ""), result)

    # Log recommendations to scorecard
    date = market_data.get("date", "")
    for side in ["top_long", "top_short"]:
        pick = result.get(side, {})
        if pick and pick.get("ticker"):
            log_recommendation(
                agent_name=agent_name,
                date=date,
                ticker=pick["ticker"],
                direction="LONG" if side == "top_long" else "SHORT",
                conviction=pick.get("conviction", 50),
            )

    return result


def run_layer2(macro_regime: dict, market_data: dict, darwinian_weights: dict) -> dict:
    """Run all 7 sector desk agents."""
    logger.info("=== Layer 2: Sector desks ===")
    picks = {}

    for agent in config.LAYER2_AGENTS:
        logger.info(f"Running {agent}")
        result = run_sector_agent(agent, macro_regime, market_data, darwinian_weights)
        picks[agent] = result

    sector_output = {
        "picks": picks,
        "date": market_data.get("date"),
        "macro_regime": macro_regime.get("regime"),
    }
    _save_state("sector_picks.json", sector_output)
    logger.info("Layer 2 complete")
    return sector_output


# --- Layer 3: Superinvestors ---

def run_superinvestor(agent_name: str, sector_picks: dict, macro_regime: dict,
                      portfolio: dict, darwinian_weights: dict, date: str) -> dict:
    """Run a single Layer 3 superinvestor agent."""
    prompt = _load_prompt(agent_name)
    weight = darwinian_weights.get(agent_name, 1.0)

    user_msg = f"""Date: {date}
Darwinian weight: {weight:.2f}

Macro Regime:
{json.dumps(macro_regime, indent=2)}

Sector Desk Recommendations:
{json.dumps(sector_picks, indent=2)}

Current Portfolio:
{json.dumps(portfolio, indent=2)}

Review portfolio and sector picks through your investment philosophy. Return ONLY valid JSON.
"""
    result = call_agent(prompt, user_msg, agent_name=agent_name)
    log_agent_output(agent_name, date, result)

    # Log any "missing_name" recommendation
    missing = result.get("missing_name", {})
    if missing and missing.get("ticker"):
        log_recommendation(
            agent_name=agent_name,
            date=date,
            ticker=missing["ticker"],
            direction="LONG",
            conviction=missing.get("conviction", 50),
        )

    return result


def run_layer3(sector_picks: dict, macro_regime: dict, portfolio: dict,
               darwinian_weights: dict, date: str) -> dict:
    """Run all 4 superinvestor agents."""
    logger.info("=== Layer 3: Superinvestors ===")
    views = {}

    for agent in config.LAYER3_AGENTS:
        logger.info(f"Running {agent}")
        result = run_superinvestor(agent, sector_picks, macro_regime, portfolio, darwinian_weights, date)
        views[agent] = result

    superinvestor_output = {
        "views": views,
        "date": date,
    }
    _save_state("superinvestor_views.json", superinvestor_output)
    logger.info("Layer 3 complete")
    return superinvestor_output


# --- Layer 4: Decision ---

def run_cro(all_outputs: dict, portfolio: dict, darwinian_weights: dict, date: str) -> dict:
    """Run the CRO (adversarial risk review)."""
    prompt = _load_prompt("cro")

    user_msg = f"""Date: {date}
Darwinian weight: {darwinian_weights.get('cro', 1.0):.2f}

All Agent Outputs:
{json.dumps(all_outputs, indent=2)}

Current Portfolio:
{json.dumps(portfolio, indent=2)}

Identify all risks, correlations, and reasons NOT to act on each idea. Return ONLY valid JSON.
"""
    result = call_agent(prompt, user_msg, agent_name="cro")
    log_agent_output("cro", date, result)
    _save_state("cro_review.json", result)
    return result


def run_alpha_discovery(all_outputs: dict, portfolio: dict, darwinian_weights: dict, date: str) -> dict:
    """Find names not mentioned by other agents."""
    prompt = _load_prompt("alpha_discovery")

    user_msg = f"""Date: {date}
Darwinian weight: {darwinian_weights.get('alpha_discovery', 1.0):.2f}

All Prior Agent Outputs:
{json.dumps(all_outputs, indent=2)}

Current Portfolio:
{json.dumps(portfolio, indent=2)}

Find high-conviction ideas not yet mentioned. Return ONLY valid JSON.
"""
    result = call_agent(prompt, user_msg, agent_name="alpha_discovery")
    log_agent_output("alpha_discovery", date, result)

    if result.get("ticker"):
        log_recommendation(
            agent_name="alpha_discovery",
            date=date,
            ticker=result["ticker"],
            direction=result.get("direction", "LONG"),
            conviction=result.get("conviction", 50),
        )
    return result


def run_cio(all_outputs: dict, cro_review: dict, portfolio: dict,
            darwinian_weights: dict, date: str) -> dict:
    """CIO makes the final portfolio decisions."""
    prompt = _load_prompt("cio")

    # Build weighted summary for CIO input
    weighted_summary = {
        "date": date,
        "darwinian_weights": darwinian_weights,
        "all_agent_outputs": all_outputs,
        "cro_review": cro_review,
        "current_portfolio": portfolio,
    }

    user_msg = f"""Date: {date}
Your Darwinian weight: {darwinian_weights.get('cio', 1.0):.2f}

All inputs (agents weighted by Darwinian scores):
{json.dumps(weighted_summary, indent=2)}

Make final BUY/SELL/HOLD decisions. Return ONLY valid JSON with portfolio_actions list.
"""
    result = call_agent(prompt, user_msg, agent_name="cio")
    log_agent_output("cio", date, result)
    _save_state("portfolio_actions.json", result)
    logger.info(f"CIO decisions: {len(result.get('actions', result.get('portfolio_actions', [])))} actions")
    return result


def run_full_cycle(market_data: dict, portfolio: dict) -> dict:
    """
    Run the full 4-layer EOD cycle.
    Returns the CIO's portfolio actions.
    """
    date = market_data.get("date", "")
    logger.info(f"=== Starting EOD cycle for {date} ===")

    darwinian_weights = load_darwinian_weights()

    # Layer 1: Macro
    macro_regime = run_layer1(market_data, darwinian_weights)

    # Layer 2: Sector desks
    sector_picks = run_layer2(macro_regime, market_data, darwinian_weights)

    # Layer 3: Superinvestors
    superinvestor_views = run_layer3(sector_picks, macro_regime, portfolio, darwinian_weights, date)

    # Layer 4a: CRO adversarial review
    all_prior = {
        "macro": macro_regime,
        "sectors": sector_picks,
        "superinvestors": superinvestor_views,
    }
    cro_review = run_cro(all_prior, portfolio, darwinian_weights, date)

    # Layer 4b: Alpha discovery
    all_with_cro = {**all_prior, "cro": cro_review}
    alpha = run_alpha_discovery(all_with_cro, portfolio, darwinian_weights, date)

    # Layer 4c: CIO final decision
    all_final = {**all_with_cro, "alpha_discovery": alpha}
    cio_decisions = run_cio(all_final, cro_review, portfolio, darwinian_weights, date)

    logger.info(f"=== EOD cycle complete for {date} ===")
    return cio_decisions
