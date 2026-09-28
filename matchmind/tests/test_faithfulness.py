import json

from matchmind.evaluation.evaluator import compute_concept_faithfulness_llm
from matchmind.schema.match_state import EvaluationReport, TacticalAdvice

RETRIEVED = [{"title": "Third-Man Run", "definition": "A third player runs onto a lay-off.", "content": "..."}]


def _advice(cited=("Third-Man Run",)):
    return TacticalAdvice(recommended_action="Pass to 8", reasoning="Use a third-man run via 8.",
                          confidence=0.8, evidence=[], cited_concepts=list(cited), alternatives=[])


def test_faithful_verdict():
    def llm(prompt, **kw):
        return json.dumps({"faithful": True, "issues": []}), 5

    assert compute_concept_faithfulness_llm(_advice(), RETRIEVED, llm_call=llm) is True


def test_unfaithful_verdict_and_prompt_contains_definition():
    seen = {}

    def llm(prompt, **kw):
        seen["p"] = prompt
        return json.dumps({"faithful": False, "issues": ["misuses concept"]}), 5

    assert compute_concept_faithfulness_llm(_advice(), RETRIEVED, llm_call=llm) is False
    assert "A third player runs onto a lay-off." in seen["p"]


def test_no_citations_or_broken_llm_returns_none():
    assert compute_concept_faithfulness_llm(_advice(cited=()), RETRIEVED, llm_call=None) is None

    def boom(prompt, **kw):
        raise RuntimeError("429")

    assert compute_concept_faithfulness_llm(_advice(), RETRIEVED, llm_call=boom) is None


def test_report_has_v1_fields():
    r = EvaluationReport(n_cases=1, citation_validity_rate=1.0, groundedness_score=1.0,
                         concept_faithfulness_rate=0.5, golden_hit_at_3=0.9, golden_ndcg_at_5=0.7,
                         retrieval_strategy="hybrid")
    assert r.concept_faithfulness_rate == 0.5 and r.retrieval_strategy == "hybrid"
