"""
Run Evaluation Script — CI-ready 5-dimension evaluation.

Dims: 0 Retrieval Quality · 1 Agreement · 2 Groundedness · 3 Actionability · 4 Consistency

Usage:
    python -m matchmind.scripts.run_evaluation
    python -m matchmind.scripts.run_evaluation --n-cases 25
    python -m matchmind.scripts.run_evaluation --from-cache        # use data_cache/
    python -m matchmind.scripts.run_evaluation --synthetic --n-cases 5
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

app = typer.Typer()
console = Console()


@app.command()
def main(
    n_cases: int = typer.Option(25, "--n-cases", help="Number of test cases"),
    synthetic: bool = typer.Option(False, "--synthetic", help="Use synthetic test cases"),
    from_cache: bool = typer.Option(False, "--from-cache", help="Use data_cache/statsbomb_testcases.json"),
    retrieval_k: int = typer.Option(3, "--retrieval-k", help="k for precision/recall@k"),
    no_llm_actionability: bool = typer.Option(False, "--no-llm", help="Use rule-based actionability scoring"),
    output: str = typer.Option("", "--output", help="Output JSON path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run MatchMind 5-dimension evaluation and save report."""

    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    console.print("\n[bold green]🔬 MatchMind Evaluation Framework[/bold green]")

    # ── Load test cases ────────────────────────────────────────
    if synthetic:
        from matchmind.tests.fixtures.sample_data import make_sample_test_cases
        test_cases = make_sample_test_cases(n=n_cases)
        console.print(f"✅ Using {len(test_cases)} synthetic test cases")
    elif from_cache:
        from matchmind.scripts.download_data import load_cached_testcases
        test_cases = load_cached_testcases()[:n_cases]
        console.print(f"✅ Loaded {len(test_cases)} cached StatsBomb test cases")
    else:
        try:
            from matchmind.data_sources.statsbomb_adapter import StatsBombAdapter
            adapter = StatsBombAdapter()
            console.print(f"Loading {n_cases} StatsBomb test cases...")
            test_cases = adapter.load_test_cases(n=n_cases)
            console.print(f"✅ Loaded {len(test_cases)} test cases from StatsBomb")
        except Exception as exc:
            console.print(f"[yellow]StatsBomb failed ({exc}), using synthetic[/yellow]")
            from matchmind.tests.fixtures.sample_data import make_sample_test_cases
            test_cases = make_sample_test_cases(n=n_cases)

    if not test_cases:
        console.print("[bold red]No test cases available![/bold red]")
        raise typer.Exit(1)

    # ── Run evaluation ─────────────────────────────────────────
    from matchmind.agent.graph import TacticalAgent
    from matchmind.evaluation.evaluator import MatchMindEvaluator

    agent = TacticalAgent()
    evaluator = MatchMindEvaluator(
        use_llm_for_actionability=not no_llm_actionability,
        retrieval_k=retrieval_k,
    )

    console.print(f"\nRunning evaluation on {len(test_cases)} cases...")
    report = evaluator.run_evaluation(test_cases, agent, n_cases=n_cases)

    # ── Display report ─────────────────────────────────────────
    table = Table(title="📊 MatchMind Evaluation Report", border_style="green")
    table.add_column("Dimension", style="bold cyan", width=25)
    table.add_column("Metric", style="yellow")
    table.add_column("Value", style="bold green", justify="right")

    def fmt_pct(v): return f"{v*100:.1f}%" if v is not None else "N/A"
    def fmt_num(v, d=2): return f"{v:.{d}f}" if v is not None else "N/A"

    table.add_row("0. Retrieval Quality", f"Precision@{report.retrieval_k or 3}", fmt_pct(report.precision_at_k))
    table.add_row("", f"Recall@{report.retrieval_k or 3}", fmt_pct(report.recall_at_k))
    table.add_row("", "Mean retrieval confidence", fmt_num(report.retrieval_confidence_mean, 3))
    table.add_row("", "Re-retrieve rate", fmt_pct(report.reretrieve_rate))
    table.add_section()
    table.add_row("1. Agreement", "Agreement Rate", fmt_pct(report.agreement_rate))
    table.add_section()
    table.add_row("2. Groundedness", "Citation Validity", fmt_pct(report.citation_validity_rate))
    table.add_row("", "Groundedness Score", fmt_pct(report.groundedness_score))
    table.add_section()
    table.add_row("3. Actionability", "Actionability Score", f"{report.actionability_score:.2f}/5.0" if report.actionability_score else "N/A")
    table.add_section()
    table.add_row("4. Consistency", "Quantitative Alignment", fmt_pct(report.quantitative_alignment))
    table.add_section()
    table.add_row("Performance", "Avg Latency", f"{report.avg_latency_ms:.0f}ms" if report.avg_latency_ms else "N/A")
    table.add_row("", "p95 Latency", f"{report.p95_latency_ms:.0f}ms" if report.p95_latency_ms else "N/A")
    table.add_row("", "Avg Tokens", str(report.avg_tokens_used) if report.avg_tokens_used else "N/A")
    table.add_row("", "Model", report.model_used or "N/A")
    table.add_row("", "N Cases", str(report.n_cases))

    console.print(table)

    # ── Save report ────────────────────────────────────────────
    from matchmind.config import settings
    output_path = Path(output) if output else settings.evaluation_output_dir / "report.json"
    evaluator.save_report(report, output_path)
    console.print(f"\n✅ Report saved to: [cyan]{output_path}[/cyan]")


if __name__ == "__main__":
    app()
