"""
KB lint — cheap consistency checks over knowledge_base/concepts/*.json.

Errors (exit 1): duplicate titles (the validator matches citations by title).
Warnings (exit 0): related_concepts pointing at unknown ids, phase that
contradicts the category, a phase_of_play_* trigger that contradicts the
concept's phase, trigger tokens no predicate understands (they can never help
condition retrieval), ingested concepts without a source.

Usage:
    python -m matchmind.scripts.kb_lint
"""
from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import typer
from rich.console import Console

from matchmind.schema.match_state import TacticalConcept

app = typer.Typer()
console = Console()

_IN_POSSESSION_PHASES = {"possession", "attacking_transition"}
_OUT_OF_POSSESSION_PHASES = {"out_of_possession", "defensive_transition"}
_TRIGGER_PHASE = {
    "phase_of_play_attacking_transition": "attacking_transition",
    "phase_of_play_defensive_transition": "defensive_transition",
}


@dataclass(frozen=True)
class LintIssue:
    level: Literal["error", "warning"]
    concept_id: str
    message: str


def _known_trigger_tokens() -> set[str]:
    try:
        from matchmind.retrieval.triggers import TRIGGER_PREDICATES
    except ImportError:  # before Task 7 moves the predicates
        from matchmind.evaluation.retrieval_quality import _TRIGGERS as TRIGGER_PREDICATES
    return set(TRIGGER_PREDICATES)


def lint_concepts(concepts: list[TacticalConcept]) -> list[LintIssue]:
    issues: list[LintIssue] = []
    ids = {c.concept_id for c in concepts}
    known_triggers = _known_trigger_tokens()

    title_counts = Counter(c.title.strip().lower() for c in concepts)
    for c in concepts:
        if title_counts[c.title.strip().lower()] > 1:
            issues.append(LintIssue("error", c.concept_id, f"duplicate title {c.title!r}"))
        for rid in c.related_concepts:
            if rid not in ids:
                issues.append(LintIssue("warning", c.concept_id, f"related concept {rid!r} does not exist"))
        if c.category == "defending" and c.phase in _IN_POSSESSION_PHASES:
            issues.append(LintIssue("warning", c.concept_id,
                                    f"defending concept has in-possession phase {c.phase!r}"))
        if c.category == "attacking" and c.phase in _OUT_OF_POSSESSION_PHASES:
            issues.append(LintIssue("warning", c.concept_id,
                                    f"attacking concept has out-of-possession phase {c.phase!r}"))
        for token, phase in _TRIGGER_PHASE.items():
            if token in c.trigger_conditions and c.phase not in (phase, "general"):
                issues.append(LintIssue("warning", c.concept_id,
                                        f"trigger {token!r} contradicts phase {c.phase!r}"))
        unknown = [t for t in c.trigger_conditions if t not in known_triggers]
        if unknown:
            issues.append(LintIssue("warning", c.concept_id, f"unchecked trigger tokens {unknown}"))
        # Hand-written concepts are the author's own words — fine without a source.
        if c.source is None and c.concept_id.startswith(("wiki_", "yt_", "fifa_")):
            issues.append(LintIssue("warning", c.concept_id, "ingested concept without source"))
    return issues


@app.command()
def main() -> None:
    from matchmind.knowledge_base.embedder import load_all_concepts

    issues = lint_concepts(load_all_concepts())
    for i in issues:
        colour = "red" if i.level == "error" else "yellow"
        console.print(f"[{colour}]{i.level.upper():7}[/{colour}] {i.concept_id}: {i.message}")
    n_err = sum(i.level == "error" for i in issues)
    console.print(f"{n_err} error(s), {len(issues) - n_err} warning(s)")
    raise typer.Exit(1 if n_err else 0)


if __name__ == "__main__":
    app()
