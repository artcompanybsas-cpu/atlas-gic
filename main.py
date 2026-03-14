"""
ATLAS - Self-Improving AI Trading Agents
Main entry point.

Usage:
    # Run live (single EOD cycle for today)
    python main.py live

    # Run backtest over a date range
    python main.py backtest --start 2024-09-01 --end 2026-03-07

    # Resume an interrupted backtest
    python main.py backtest --start 2024-09-01 --end 2026-03-07 --resume

    # Show current agent weights and scores
    python main.py status
"""
import sys
import json
import argparse
from datetime import datetime
from rich.console import Console
from rich.table import Table

console = Console()


def cmd_live() -> None:
    """Run one EOD cycle for today."""
    from src.agents.market_data import get_market_snapshot
    from src.agents.eod_cycle import run_full_cycle
    from src.agents.backtest_loop import _load_portfolio, _mark_portfolio_to_market
    from src.agents.scorecard import update_darwinian_weights, update_forward_returns
    from src.agents.autoresearch import maybe_run_autoresearch

    today = datetime.now().strftime("%Y-%m-%d")
    console.print(f"\n[bold green]ATLAS Live Mode[/bold green] — {today}\n")

    portfolio = _load_portfolio()
    console.print(f"Portfolio value: ${portfolio['total_value']:,.0f}")

    update_forward_returns(today)
    market_data = get_market_snapshot(today)
    cio_decisions = run_full_cycle(market_data, portfolio)

    console.print("\n[bold]CIO Decisions:[/bold]")
    actions = cio_decisions.get("actions") or cio_decisions.get("portfolio_actions", [])
    for action in actions:
        console.print(f"  {action.get('action', '?')} {action.get('ticker', '?')} x{action.get('shares', 0)} — {action.get('rationale', '')[:80]}")

    update_darwinian_weights(today)
    # Use a rough day counter (days since 2024-09-01)
    day_num = (datetime.now() - datetime(2024, 9, 1)).days
    maybe_run_autoresearch(today, day_num)


def cmd_backtest(start: str, end: str, resume: bool = False) -> None:
    """Run backtest."""
    from src.agents.backtest_loop import run_backtest

    console.print(f"\n[bold green]ATLAS Backtest[/bold green]")
    console.print(f"Period: {start} → {end}")
    console.print(f"Resume: {resume}\n")

    result = run_backtest(start, end, resume=resume)

    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  Trading days: {result['trading_days']}")
    console.print(f"  Starting value: ${result['starting_value']:,.0f}")
    console.print(f"  Ending value: ${result['ending_value']:,.0f}")
    console.print(f"  Total return: {result['total_return_pct']:.1f}%")


def cmd_status() -> None:
    """Show current agent weights and recent performance."""
    from src.agents.scorecard import load_darwinian_weights, get_all_agent_sharpes
    import config

    weights = load_darwinian_weights()
    sharpes = get_all_agent_sharpes(lookback_days=60)

    table = Table(title="ATLAS Agent Status")
    table.add_column("Agent", style="cyan")
    table.add_column("Layer", justify="center")
    table.add_column("Darwinian Weight", justify="right")
    table.add_column("Sharpe (60d)", justify="right")

    layer_names = {1: "Macro", 2: "Sector", 3: "Superinvestor", 4: "Decision"}

    for agent, info in config.AGENT_DEFINITIONS.items():
        layer = info["layer"]
        weight = weights.get(agent, 1.0)
        sharpe = sharpes.get(agent)
        sharpe_str = f"{sharpe:.3f}" if sharpe is not None else "N/A"

        weight_style = "green" if weight >= 2.0 else "red" if weight <= 0.4 else "white"
        table.add_row(
            agent,
            layer_names[layer],
            f"[{weight_style}]{weight:.2f}[/{weight_style}]",
            sharpe_str,
        )

    console.print(table)

    # Autoresearch summary
    from config import DATA_DIR
    ar_log = DATA_DIR / "track_record" / "autoresearch_log.json"
    if ar_log.exists():
        with open(ar_log) as f:
            log = json.load(f)
        kept = sum(1 for e in log if e.get("kept"))
        console.print(f"\n[bold]Autoresearch:[/bold] {len(log)} attempts, {kept} kept ({100*kept//max(len(log),1)}% success)")


def main() -> None:
    parser = argparse.ArgumentParser(description="ATLAS - Self-Improving AI Trading Agents")
    subparsers = parser.add_subparsers(dest="command")

    # live
    subparsers.add_parser("live", help="Run one EOD cycle for today")

    # backtest
    bt = subparsers.add_parser("backtest", help="Run historical backtest")
    bt.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    bt.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    bt.add_argument("--resume", action="store_true", help="Resume from last state")

    # status
    subparsers.add_parser("status", help="Show agent weights and performance")

    args = parser.parse_args()

    if args.command == "live":
        cmd_live()
    elif args.command == "backtest":
        cmd_backtest(args.start, args.end, resume=args.resume)
    elif args.command == "status":
        cmd_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
