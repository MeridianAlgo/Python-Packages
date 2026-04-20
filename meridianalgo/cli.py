"""
MeridianAlgo CLI

Command-line interface for the MeridianAlgo quantitative finance library.
"""

from __future__ import annotations

import argparse
import os
import sys


def cmd_version(args: argparse.Namespace) -> None:
    from meridianalgo import __version__

    print(f"meridianalgo {__version__}")


def cmd_info(args: argparse.Namespace) -> None:
    os.environ.setdefault("MERIDIANALGO_QUIET", "1")
    from meridianalgo import ModuleRegistry, __version__

    print(f"MeridianAlgo {__version__}")
    print()
    print("Module availability:")
    for module, available in ModuleRegistry.status().items():
        status = "OK" if available else "unavailable"
        print(f"  {module:<20} {status}")


def cmd_demo(args: argparse.Namespace) -> None:
    os.environ.setdefault("MERIDIANALGO_QUIET", "1")
    import numpy as np
    import pandas as pd

    print("MeridianAlgo Demo — Portfolio Optimization")
    print("=" * 50)

    np.random.seed(42)
    n_days = 252
    assets = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    returns = pd.DataFrame(
        np.random.randn(n_days, len(assets)) * 0.015,
        columns=assets,
    )

    from meridianalgo import (
        HierarchicalRiskParity,
        calculate_max_drawdown,
        calculate_sharpe_ratio,
        calculate_sortino_ratio,
    )

    hrp = HierarchicalRiskParity()
    weights = hrp.optimize(returns)
    portfolio_returns = (returns * weights).sum(axis=1)

    print(f"\nHRP Weights:")
    for asset, w in weights.items():
        print(f"  {asset}: {w:.1%}")

    sharpe = calculate_sharpe_ratio(portfolio_returns)
    sortino = calculate_sortino_ratio(portfolio_returns)
    max_dd = calculate_max_drawdown(portfolio_returns)

    print(f"\nPortfolio Metrics (annualized):")
    print(f"  Sharpe Ratio:  {sharpe:.3f}")
    print(f"  Sortino Ratio: {sortino:.3f}")
    print(f"  Max Drawdown:  {max_dd:.2%}")


def cmd_metrics(args: argparse.Namespace) -> None:
    """Calculate metrics for a ticker from Yahoo Finance."""
    os.environ.setdefault("MERIDIANALGO_QUIET", "1")

    ticker = args.ticker
    period = args.period

    print(f"Fetching {ticker} ({period})...")

    try:
        import yfinance as yf

        data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if data.empty:
            print(f"No data found for {ticker}")
            sys.exit(1)

        close = data["Close"].squeeze()
        returns = close.pct_change().dropna()

        from meridianalgo import (
            calculate_calmar_ratio,
            calculate_max_drawdown,
            calculate_sharpe_ratio,
            calculate_sortino_ratio,
        )

        sharpe = calculate_sharpe_ratio(returns)
        sortino = calculate_sortino_ratio(returns)
        calmar = calculate_calmar_ratio(returns)
        max_dd = calculate_max_drawdown(returns)
        ann_return = returns.mean() * 252
        ann_vol = returns.std() * (252**0.5)

        print(f"\n{ticker} Performance Metrics")
        print("=" * 40)
        print(f"  Annualized Return:  {ann_return:.2%}")
        print(f"  Annualized Vol:     {ann_vol:.2%}")
        print(f"  Sharpe Ratio:       {sharpe:.3f}")
        print(f"  Sortino Ratio:      {sortino:.3f}")
        print(f"  Calmar Ratio:       {calmar:.3f}")
        print(f"  Max Drawdown:       {max_dd:.2%}")
        print(f"  Observations:       {len(returns)}")

    except ImportError:
        print("yfinance required: pip install yfinance")
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meridianalgo",
        description="MeridianAlgo — Quantitative Finance Toolkit",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s"
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    subparsers.add_parser("version", help="Print version and exit").set_defaults(
        func=cmd_version
    )
    subparsers.add_parser(
        "info", help="Show module availability"
    ).set_defaults(func=cmd_info)
    subparsers.add_parser(
        "demo", help="Run a quick portfolio optimization demo"
    ).set_defaults(func=cmd_demo)

    metrics_p = subparsers.add_parser(
        "metrics", help="Compute performance metrics for a ticker"
    )
    metrics_p.add_argument("ticker", help="Ticker symbol (e.g. AAPL)")
    metrics_p.add_argument(
        "--period", default="2y", help="yfinance period (default: 2y)"
    )
    metrics_p.set_defaults(func=cmd_metrics)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
