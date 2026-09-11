"""
Run Evaluation Script - CI-ready 5-dimension evaluation.

Dims: 0 Retrieval Quality | 1 Agreement | 2 Groundedness | 3 Actionability | 4 Consistency

Usage:
    python -m matchmind.scripts.run_evaluation --synthetic --n-cases 5 --no-llm
    python -m matchmind.scripts.run_evaluation --n-cases 25
    python -m matchmind.scripts.run_evaluation --from-cache            # data_cache/
    python -m matchmind.scripts.run_evaluation --consistency-runs 3    # + Dim 4 multi-run
"""
from __future__ import annotations

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
    n_cases: int = typer.Option(25, help="Number of test cases"),
    synthetic: bool = typer.Option(False, "--synthetic/--no-synthetic", help="Use synthetic test cases"),
    from_cache: bool = typer.Option(
        False, "--from-cache/--no-from-cache", help="Use data_cache/statsbomb_testcases.json"
    ),
    retrieval_k: int = typer.Option(3, help="k for precision/recall@k"),
    consistency_runs: int = typer.Option(
        0, help="If >0, re-run the first few snapshots N times for the Dim 4 consistency rate"
    ),
    no_llm_actionability: bool = typer.Option(
        False, "--no-llm/--llm", help="Rule-based actionability + skip the groundedness LLM judge"
    ),
    output: str = typer.Option("", help="Output JSON path"),
    verbose: bool = typer.Option(False, "--verbose/--no-verbose", help="Enable debug logging"),
) -> None:
    """Run MatchMind 5-dimension evaluation and save the report."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    console.print("\n[bold green]MatchMind Evaluation Framework[/bold green]")

    if synthetic and from_cache:
        console.print("[yellow]--synthetic and --from-cache both set; using --synthetic.[/yellow]")
        from_cache = False

    # ── Load test cases ────────────────────────────────────────
    test_cases = _load_cases(synthetic, from_cache, n_cases)
    if not test_cases:
        console.print("[bold red]No test cases available![/bold red]")
        raise typer.Exit(1)

    # ── Build agent + evaluator ───────────────────────────────
    try:
        from matchmind.agent.graph import TacticalAgent
        from matchmind.evaluation.evaluator import MatchMindEvaluator
    except ImportError as exc:
        console.print(f"[bold red]Missing dependency: {exc}[/bold red]  → run `poetry install`")
        raise typer.Exit(1)

    try:
        agent = TacticalAgent()
    except Exception as exc:
        console.print(f"[bold red]Could not initialise the agent:[/bold red] {exc}")
        console.print("[yellow]Check GEMINI_API_KEY in .env and that the KB is built (build_kb).[/yellow]")
        raise typer.Exit(1)

    evaluator = MatchMindEvaluator(
        use_llm_for_actionability=not no_llm_actionability,
        retrieval_k=retrieval_k,
    )

    console.print(f"\nRunning evaluation on {len(test_cases)} cases (k={retrieval_k})...")
    report = evaluator.run_evaluation(test_cases, agent, n_cases=n_cases)

    if consistency_runs > 0:
        console.print(f"Dim 4: re-running first snapshots x{consistency_runs} for consistency...")
        cons = evaluator.run_consistency(test_cases, agent, n_runs=consistency_runs)
        report = report.model_copy(update={"consistency_rate": cons.get("consistency_rate")})

    _render_report(report)

    from matchmind.config import settings

    output_path = Path(output) if output else settings.evaluation_output_dir / "report.json"
    evaluator.save_report(report, output_path)
    console.print(f"\n[green]Report saved to:[/green] [cyan]{output_path}[/cyan]")


# ──────────────────────────────────────────────────────────────────────────────


def _load_cases(synthetic: bool, from_cache: bool, n_cases: int) -> list[dict]:
    from matchmind.tests.fixtures.sample_data import make_sample_test_cases

    if synthetic:
        cases = make_sample_test_cases(n=n_cases)
        console.print(f"[green]Using {len(cases)} synthetic test cases[/green]")
        return cases

    if from_cache:
        try:
            from matchmind.scripts.download_data import load_cached_testcases

            cases = load_cached_testcases()[:n_cases]
            console.print(f"[green]Loaded {len(cases)} cached StatsBomb test cases[/green]")
            return cases
        except Exception as exc:
            console.print(f"[yellow]Cache load failed ({exc}); falling back to synthetic[/yellow]")
            return make_sample_test_cases(n=n_cases)

    try:
        from matchmind.data_sources.statsbomb_adapter import StatsBombAdapter

        console.print(f"Loading {n_cases} StatsBomb test cases (network)...")
        cases = StatsBombAdapter().load_test_cases(n=n_cases)
        console.print(f"[green]Loaded {len(cases)} test cases from StatsBomb[/green]")
        return cases
    except Exception as exc:
        console.print(f"[yellow]StatsBomb failed ({exc}); using synthetic[/yellow]")
        return make_sample_test_cases(n=n_cases)


def _render_report(report) -> None:
    def pct(v):
        return f"{v * 100:.1f}%" if v is not None else "N/A"

    def num(v, d=3):
        return f"{v:.{d}f}" if v is not None else "N/A"

    k = report.retrieval_k or 3
    t = Table(title="MatchMind Evaluation Report", border_style="green")
    t.add_column("Dimension", style="bold cyan", width=24)
    t.add_column("Metric", style="yellow")
    t.add_column("Value", style="bold green", justify="right")

    t.add_row("0. Retrieval Quality", f"Precision@{k}", pct(report.precision_at_k))
    t.add_row("", f"Recall@{k}", pct(report.recall_at_k))
    t.add_row("", "Mean retrieval confidence", num(report.retrieval_confidence_mean))
    t.add_row("", "Re-retrieve rate", pct(report.reretrieve_rate))
    t.add_section()
    t.add_row("1. Agreement", "Agreement rate (reference only)", pct(report.agreement_rate))
    t.add_section()
    t.add_row("2. Groundedness", "Citation validity", pct(report.citation_validity_rate))
    t.add_row("", "Logical consistency (LLM judge)", pct(report.logical_consistency_rate))
    t.add_row("", "Groundedness score", pct(report.groundedness_score))
    t.add_section()
    a = report.actionability_score
    t.add_row("3. Actionability", "Score (1-5)", f"{a:.2f}/5.0" if a is not None else "N/A")
    t.add_section()
    t.add_row("4. Consistency", "Same action across runs", pct(report.consistency_rate))
    t.add_row("", "Quantitative alignment", pct(report.quantitative_alignment))
    t.add_section()
    t.add_row("Performance", "Avg latency", f"{report.avg_latency_ms:.0f}ms" if report.avg_latency_ms else "N/A")
    t.add_row("", "p95 latency", f"{report.p95_latency_ms:.0f}ms" if report.p95_latency_ms else "N/A")
    t.add_row("", "Avg tokens", str(report.avg_tokens_used) if report.avg_tokens_used else "N/A")
    t.add_row("", "Model", report.model_used or "N/A")
    t.add_row("", "N cases", str(report.n_cases))

    console.print(t)


if __name__ == "__main__":
    app()
