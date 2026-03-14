"""
Git operations for ATLAS autoresearch.
Handles branch creation, commits, merges, and reverts for prompt evolution.
"""
import subprocess
from pathlib import Path
from src.utils.logging import get_logger

logger = get_logger(__name__)
REPO_ROOT = Path(__file__).parent.parent.parent


def run_git(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a git command in the repo root."""
    result = subprocess.run(
        ["git"] + args,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        logger.error(f"Git command failed: git {' '.join(args)}\n{result.stderr}")
        raise RuntimeError(f"Git error: {result.stderr}")
    return result


def current_branch() -> str:
    return run_git(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()


def create_autoresearch_branch(agent_name: str, modification_slug: str) -> str:
    """Create a feature branch for a prompt modification trial."""
    branch = f"autoresearch/{agent_name}-{modification_slug}"
    run_git(["checkout", "-b", branch])
    logger.info(f"Created autoresearch branch: {branch}")
    return branch


def commit_prompt_modification(agent_name: str, description: str) -> str:
    """Stage and commit changes to an agent's prompt file."""
    prompt_glob = f"src/prompts/**/{agent_name}.md"
    run_git(["add", "-A", "--", f"src/prompts/"])
    result = run_git(["commit", "-m", f"autoresearch: {agent_name} - {description}"])
    commit_hash = run_git(["rev-parse", "HEAD"]).stdout.strip()
    logger.info(f"Committed prompt modification: {commit_hash[:8]}")
    return commit_hash


def merge_autoresearch_branch(feature_branch: str, base_branch: str = "master") -> bool:
    """Merge a successful autoresearch branch back to base."""
    try:
        run_git(["checkout", base_branch])
        run_git(["merge", "--no-ff", feature_branch, "-m", f"merge: {feature_branch} (autoresearch success)"])
        run_git(["branch", "-d", feature_branch])
        logger.info(f"Merged {feature_branch} into {base_branch}")
        return True
    except RuntimeError as e:
        logger.error(f"Merge failed: {e}")
        return False


def revert_autoresearch_branch(feature_branch: str, base_branch: str = "master") -> bool:
    """Discard a failed autoresearch branch."""
    try:
        run_git(["checkout", base_branch])
        run_git(["branch", "-D", feature_branch])
        logger.info(f"Reverted (deleted) {feature_branch}")
        return True
    except RuntimeError as e:
        logger.error(f"Revert failed: {e}")
        return False


def get_prompt_history(agent_name: str, n: int = 10) -> list[dict]:
    """Return last n commits that touched an agent's prompt."""
    result = run_git([
        "log", f"-{n}", "--oneline", "--follow", "--",
        f"src/prompts/layer1/{agent_name}.md",
        f"src/prompts/layer2/{agent_name}.md",
        f"src/prompts/layer3/{agent_name}.md",
        f"src/prompts/layer4/{agent_name}.md",
    ], check=False)
    commits = []
    for line in result.stdout.strip().splitlines():
        if line:
            parts = line.split(" ", 1)
            commits.append({"hash": parts[0], "message": parts[1] if len(parts) > 1 else ""})
    return commits
