"""
Anthropic API wrapper for ATLAS agents.
Handles retries, rate limiting, and structured JSON responses.
"""
import json
import time
import re
from typing import Optional
import anthropic
import config
from src.utils.logging import get_logger

logger = get_logger(__name__)

_client: Optional[anthropic.Anthropic] = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


def call_agent(
    system_prompt: str,
    user_message: str,
    agent_name: str = "unknown",
    max_retries: int = 3,
    expect_json: bool = True,
) -> dict | str:
    """
    Call an ATLAS agent via the Anthropic API.
    Returns parsed JSON dict if expect_json=True, else raw string.
    """
    client = get_client()
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=config.LLM_MODEL,
                max_tokens=config.LLM_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            content = response.content[0].text

            if not expect_json:
                return content

            return _parse_json_response(content, agent_name)

        except anthropic.RateLimitError as e:
            wait = 2 ** attempt * 5
            logger.warning(f"[{agent_name}] Rate limit hit, waiting {wait}s (attempt {attempt+1})")
            time.sleep(wait)
            last_error = e
        except anthropic.APIError as e:
            wait = 2 ** attempt * 2
            logger.warning(f"[{agent_name}] API error: {e}, retrying in {wait}s")
            time.sleep(wait)
            last_error = e
        except json.JSONDecodeError as e:
            logger.error(f"[{agent_name}] JSON parse error: {e}")
            last_error = e
            break

    logger.error(f"[{agent_name}] All {max_retries} attempts failed: {last_error}")
    return {}


def _parse_json_response(content: str, agent_name: str) -> dict:
    """Extract and parse JSON from LLM response, handling markdown code blocks."""
    # Try direct parse first
    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` blocks
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding the first { ... } block
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    logger.error(f"[{agent_name}] Could not parse JSON from response: {content[:200]}")
    return {}
