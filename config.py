"""
ATLAS Configuration
Loads environment variables and provides typed config access.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "src/data")
PROMPTS_DIR = BASE_DIR / os.getenv("PROMPTS_DIR", "src/prompts")
LOG_DIR = BASE_DIR / os.getenv("LOG_DIR", "logs")

# Ensure directories exist
for d in [DATA_DIR / "state", DATA_DIR / "backtest", DATA_DIR / "track_record", LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
FMP_API_KEY = os.getenv("FMP_API_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# Portfolio Settings
STARTING_CAPITAL = float(os.getenv("STARTING_CAPITAL", "1000000"))
MAX_POSITION_SIZE_PCT = float(os.getenv("MAX_POSITION_SIZE_PCT", "0.15"))
MAX_GROSS_EXPOSURE = float(os.getenv("MAX_GROSS_EXPOSURE", "1.5"))
MAX_NET_EXPOSURE = float(os.getenv("MAX_NET_EXPOSURE", "0.8"))

# Autoresearch Settings
AUTORESEARCH_WINDOW_DAYS = int(os.getenv("AUTORESEARCH_WINDOW_DAYS", "5"))
AUTORESEARCH_MIN_RECOMMENDATIONS = int(os.getenv("AUTORESEARCH_MIN_RECOMMENDATIONS", "10"))
DARWINIAN_WEIGHT_CEILING = float(os.getenv("DARWINIAN_WEIGHT_CEILING", "2.5"))
DARWINIAN_WEIGHT_FLOOR = float(os.getenv("DARWINIAN_WEIGHT_FLOOR", "0.3"))
DARWINIAN_REWARD_MULTIPLIER = float(os.getenv("DARWINIAN_REWARD_MULTIPLIER", "1.05"))
DARWINIAN_PENALTY_MULTIPLIER = float(os.getenv("DARWINIAN_PENALTY_MULTIPLIER", "0.95"))

# LLM Settings
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# Agent definitions (name -> layer, type)
AGENT_DEFINITIONS = {
    # Layer 1 - Macro
    "central_bank":        {"layer": 1, "type": "macro"},
    "geopolitical":        {"layer": 1, "type": "macro"},
    "china":               {"layer": 1, "type": "macro"},
    "dollar":              {"layer": 1, "type": "macro"},
    "yield_curve":         {"layer": 1, "type": "macro"},
    "commodities":         {"layer": 1, "type": "macro"},
    "volatility":          {"layer": 1, "type": "macro"},
    "emerging_markets":    {"layer": 1, "type": "macro"},
    "news_sentiment":      {"layer": 1, "type": "macro"},
    "institutional_flow":  {"layer": 1, "type": "macro"},
    # Layer 2 - Sector
    "semiconductor":       {"layer": 2, "type": "sector"},
    "energy":              {"layer": 2, "type": "sector"},
    "biotech":             {"layer": 2, "type": "sector"},
    "consumer":            {"layer": 2, "type": "sector"},
    "industrials":         {"layer": 2, "type": "sector"},
    "financials":          {"layer": 2, "type": "sector"},
    "relationship_mapper": {"layer": 2, "type": "sector"},
    # Layer 3 - Superinvestors
    "druckenmiller":       {"layer": 3, "type": "superinvestor"},
    "aschenbrenner":       {"layer": 3, "type": "superinvestor"},
    "baker":               {"layer": 3, "type": "superinvestor"},
    "ackman":              {"layer": 3, "type": "superinvestor"},
    # Layer 4 - Decision
    "cro":                 {"layer": 4, "type": "decision"},
    "alpha_discovery":     {"layer": 4, "type": "decision"},
    "autonomous_execution":{"layer": 4, "type": "decision"},
    "cio":                 {"layer": 4, "type": "decision"},
}

LAYER1_AGENTS = [k for k, v in AGENT_DEFINITIONS.items() if v["layer"] == 1]
LAYER2_AGENTS = [k for k, v in AGENT_DEFINITIONS.items() if v["layer"] == 2]
LAYER3_AGENTS = [k for k, v in AGENT_DEFINITIONS.items() if v["layer"] == 3]
LAYER4_AGENTS = [k for k, v in AGENT_DEFINITIONS.items() if v["layer"] == 4]
ALL_AGENTS = list(AGENT_DEFINITIONS.keys())

# Default Darwinian weights (from backtest final state)
DEFAULT_DARWINIAN_WEIGHTS = {
    "central_bank": 0.3,
    "geopolitical": 2.5,
    "china": 1.36,
    "dollar": 0.44,
    "yield_curve": 1.13,
    "commodities": 2.5,
    "volatility": 2.5,
    "emerging_markets": 0.35,
    "news_sentiment": 1.55,
    "institutional_flow": 0.3,
    "semiconductor": 0.3,
    "energy": 2.5,
    "biotech": 0.36,
    "consumer": 1.92,
    "industrials": 2.5,
    "financials": 0.47,
    "druckenmiller": 0.55,
    "aschenbrenner": 2.5,
    "baker": 0.46,
    "ackman": 2.5,
    "cro": 1.0,
    "alpha_discovery": 1.82,
    "autonomous_execution": 1.0,
    "cio": 0.3,
}
