"""
Autoresearch: self-improving agent prompts via Sharpe-guided evolution.
Inspired by Karpathy's autoresearch, applied to financial markets.

Loop:
1. Identify worst-performing agent (lowest Sharpe)
2. Generate ONE targeted prompt modification
3. Create git feature branch
4. Run for AUTORESEARCH_WINDOW_DAYS
5. If Sharpe improved: git merge (keep)
   Else: git reset (revert)
"""
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Optional
import config
from src.utils.llm import call_agent
from src.utils.git_ops import (
    create_autoresearch_branch,
    commit_prompt_modification,
    merge_autoresearch_branch,
    revert_autoresearch_branch,
    current_branch,
)
from src.agents.scorecard import (
    calculate_agent_sharpe,
    get_worst_agent,
    load_scorecard,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

AUTORESEARCH_LOG = config.DATA_DIR / "track_record" / "autoresearch_log.json"
AUTORESEARCH_STATE = config.DATA_DIR / "state" / "autoresearch_state.json"


def _load_autoresearch_log() -> list:
    if AUTORESEARCH_LOG.exists():
        with open(AUTORESEARCH_LOG) as f:
            return json.load(f)
    return []


def _save_autoresearch_log(log: list) -> None:
    AUTORESEARCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(AUTORESEARCH_LOG, "w") as f:
        json.dump(log, f, indent=2)


def _load_state() -> dict:
    if AUTORESEARCH_STATE.exists():
        with open(AUTORESEARCH_STATE) as f:
            return json.load(f)
    return {"active_trial": None, "recently_modified": {}}


def _save_state(state: dict) -> None:
    AUTORESEARCH_STATE.parent.mkdir(parents=True, exist_ok=True)
    with open(AUTORESEARCH_STATE, "w") as f:
        json.dump(state, f, indent=2)


def _get_agent_prompt_path(agent_name: str) -> Optional[Path]:
    for layer_dir in ["layer1", "layer2", "layer3", "layer4"]:
        path = config.PROMPTS_DIR / layer_dir / f"{agent_name}.md"
        if path.exists():
            return path
    return None


def _get_recent_recommendations(agent_name: str, n: int = 20) -> list:
    """Get the n most recent recommendations for an agent."""
    sc = load_scorecard()
    recs = [r for r in sc["recommendations"] if r["agent"] == agent_name]
    return recs[-n:]


def generate_prompt_modification(agent_name: str, current_prompt: str,
                                  recent_recs: list, current_sharpe: float) -> Optional[dict]:
    """
    Use Claude to propose ONE targeted modification to an agent's prompt.
    Returns: {"modification_description": str, "new_prompt": str}
    """
    system = """You are an AI prompt engineer specialising in financial trading systems.
Your job is to improve the performance of AI trading agents by making targeted, specific modifications to their prompts.
You will analyse recent poor recommendations and propose exactly ONE focused change.
Return a JSON object with:
- "diagnosis": what went wrong with recent recommendations
- "modification": one specific change to make (1-2 sentences)
- "new_prompt": the full updated prompt text
"""

    recs_summary = json.dumps([{
        "date": r["date"],
        "ticker": r["ticker"],
        "direction": r["direction"],
        "conviction": r["conviction"],
        "return_5d": r.get("return_5d"),
        "return_20d": r.get("return_20d"),
    } for r in recent_recs], indent=2)

    user_msg = f"""Agent: {agent_name}
Current rolling Sharpe: {current_sharpe:.3f}
Recent recommendations (last 20):
{recs_summary}

Current prompt:
---
{current_prompt}
---

Analyse the failure patterns and propose ONE targeted modification.
The modification should address the specific failure mode, not be a generic improvement.
Return ONLY valid JSON.
"""
    result = call_agent(system, user_msg, agent_name="autoresearch_generator")
    if not result or "new_prompt" not in result:
        return None
    return result


def start_autoresearch_trial(agent_name: str, date: str, day_number: int) -> bool:
    """
    Start a new autoresearch trial for the given agent.
    Returns True if trial was started successfully.
    """
    state = _load_state()

    # Don't start if there's already an active trial
    if state.get("active_trial"):
        logger.info(f"Autoresearch trial already active for {state['active_trial']['agent']}")
        return False

    # Check agent is eligible (not recently modified)
    recently_modified = state.get("recently_modified", {})
    if agent_name in recently_modified:
        last_mod_day = recently_modified[agent_name]
        if day_number - last_mod_day < config.AUTORESEARCH_WINDOW_DAYS:
            logger.info(f"{agent_name} was recently modified (day {last_mod_day}), skipping")
            return False

    # Calculate pre-modification Sharpe
    pre_sharpe = calculate_agent_sharpe(agent_name, lookback_days=60)
    if pre_sharpe is None:
        logger.info(f"Not enough data to calculate Sharpe for {agent_name}")
        return False

    # Load current prompt
    prompt_path = _get_agent_prompt_path(agent_name)
    if not prompt_path:
        logger.error(f"No prompt file found for {agent_name}")
        return False

    current_prompt = prompt_path.read_text()
    recent_recs = _get_recent_recommendations(agent_name, n=20)

    if len(recent_recs) < config.AUTORESEARCH_MIN_RECOMMENDATIONS:
        logger.info(f"Not enough recommendations for {agent_name} ({len(recent_recs)} < {config.AUTORESEARCH_MIN_RECOMMENDATIONS})")
        return False

    logger.info(f"Starting autoresearch trial for {agent_name} (Sharpe: {pre_sharpe:.3f})")

    # Generate modification
    mod_result = generate_prompt_modification(agent_name, current_prompt, recent_recs, pre_sharpe)
    if not mod_result:
        logger.error(f"Failed to generate modification for {agent_name}")
        return False

    modification_desc = mod_result.get("modification", "prompt modification")
    new_prompt = mod_result.get("new_prompt", "")

    if not new_prompt:
        return False

    # Create slug from description for branch name
    slug = re.sub(r"[^a-z0-9]+", "-", modification_desc.lower())[:40].strip("-")

    try:
        # Create git branch
        branch = create_autoresearch_branch(agent_name, slug)

        # Write modified prompt
        prompt_path.write_text(new_prompt)

        # Commit
        commit_prompt_modification(agent_name, modification_desc)

        # Record trial state
        state["active_trial"] = {
            "agent": agent_name,
            "branch": branch,
            "start_day": day_number,
            "start_date": date,
            "pre_sharpe": pre_sharpe,
            "modification": modification_desc,
            "diagnosis": mod_result.get("diagnosis", ""),
        }
        state["recently_modified"][agent_name] = day_number
        _save_state(state)

        logger.info(f"Autoresearch trial started: {branch}")
        logger.info(f"Modification: {modification_desc}")
        return True

    except Exception as e:
        logger.error(f"Failed to start autoresearch trial: {e}")
        return False


def evaluate_autoresearch_trial(date: str, day_number: int) -> Optional[dict]:
    """
    Check if the active trial has run for AUTORESEARCH_WINDOW_DAYS.
    If so, evaluate and keep or revert.
    Returns result dict if evaluated, None if trial still running.
    """
    state = _load_state()
    trial = state.get("active_trial")
    if not trial:
        return None

    elapsed = day_number - trial["start_day"]
    if elapsed < config.AUTORESEARCH_WINDOW_DAYS:
        logger.info(f"Autoresearch trial in progress: {trial['agent']} (day {elapsed}/{config.AUTORESEARCH_WINDOW_DAYS})")
        return None

    # Evaluate
    agent = trial["agent"]
    post_sharpe = calculate_agent_sharpe(agent, lookback_days=30)

    if post_sharpe is None:
        logger.warning(f"Cannot evaluate trial for {agent} - insufficient data")
        return None

    pre_sharpe = trial["pre_sharpe"]
    improved = post_sharpe > pre_sharpe
    branch = trial["branch"]
    base_branch = "master"

    logger.info(f"Autoresearch evaluation: {agent}")
    logger.info(f"  Pre-Sharpe: {pre_sharpe:.3f}")
    logger.info(f"  Post-Sharpe: {post_sharpe:.3f}")
    logger.info(f"  Improved: {improved}")

    if improved:
        success = merge_autoresearch_branch(branch, base_branch)
        action = "kept" if success else "merge_failed"
    else:
        success = revert_autoresearch_branch(branch, base_branch)
        action = "reverted" if success else "revert_failed"

    result = {
        "day": day_number,
        "date": date,
        "agent": agent,
        "modification": trial["modification"],
        "diagnosis": trial.get("diagnosis", ""),
        "pre_sharpe": pre_sharpe,
        "post_sharpe": post_sharpe,
        "kept": improved and action == "kept",
    }

    # Update log
    log = _load_autoresearch_log()
    log.append(result)
    _save_autoresearch_log(log)

    # Clear active trial
    state["active_trial"] = None
    _save_state(state)

    logger.info(f"Autoresearch result: {action} ({agent})")
    return result


def maybe_run_autoresearch(date: str, day_number: int) -> None:
    """
    Called each trading day. Evaluates any active trial and potentially starts a new one.
    Frequency: attempt to start a new trial every ~7 days if no active trial.
    """
    # First check if active trial should be evaluated
    result = evaluate_autoresearch_trial(date, day_number)
    if result:
        logger.info(f"Trial complete: {result['agent']} - {'KEPT' if result['kept'] else 'REVERTED'}")

    # Only try starting new trial periodically (every ~7 days)
    if day_number % 7 != 0:
        return

    state = _load_state()
    if state.get("active_trial"):
        return

    # Find worst agent
    recently_modified = set(
        agent for agent, mod_day in state.get("recently_modified", {}).items()
        if day_number - mod_day < config.AUTORESEARCH_WINDOW_DAYS * 2
    )
    worst = get_worst_agent(lookback_days=60, exclude_recently_modified=recently_modified)

    if worst:
        logger.info(f"Autoresearch candidate: {worst}")
        start_autoresearch_trial(worst, date, day_number)
