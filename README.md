# MatchMind

**A football tactical assistant grounded in real match data and a curated tactics knowledge base.**

Pick a moment from a real match, pick a player, and ask what they should do. MatchMind reads the
situation from player positions, retrieves the relevant tactical concepts, checks the options
with a pitch-control model, and answers in plain language, in any language. Every answer cites
the concepts it used, and a validator rejects answers that cite anything it did not retrieve.

| | |
|---|---|
| **Status** | **V1 (RAG over StatsBomb 360 snapshots): complete** (evaluated 2026-10-15). V2 (live) and V3 (computer vision) are planned, not started. See [Roadmap](#roadmap). |
| **Data** | [StatsBomb Open Data](https://github.com/statsbomb/open-data) 360 freeze frames (non-commercial, with attribution) |
| **Knowledge base** | 333 tactical concepts from hand-written notes, FIFA Training Centre, Wikipedia and YouTube tactics channels |
| **LLMs** | Free-tier friendly: a configurable failover chain across Groq, Cerebras, Gemini, Mistral and OpenRouter (single-provider Gemini by default) |

---

## Goals

1. **Advice a coach can trust.** Every recommendation is tied to the actual player positions,
   cites the tactical concepts it relies on, and is checked against a quantitative model, not
   just an LLM's opinion.
2. **Retrieval quality you can measure.** A hand-labelled golden set guards retrieval in CI, and
   an end-to-end evaluation script scores full answers. A change to the knowledge base or the
   retriever shows up as a number, not a feeling.
3. **One schema for every data source.** Event data with freeze frames today, a live event
   stream in V2, positions from video in V3. Everything is converted into the same `MatchState`,
   so the reasoning stack never changes when the source does.
4. **Cheap to run.** The system works on free-tier LLMs. Retrieval, pitch control, the Example
   Library and the "what happens next?" projection use no LLM tokens. Providers fail over
   automatically when a quota runs out.
5. **Speaks the user's language.** Answers read like a coach talking, in whatever language the
   question was asked, with the numbers kept in a separate "why" section.

## What it does

- **Situation analysis from positions.** About 25 spatial and structural features per snapshot:
  pressure on the ball, open passing lanes, defensive-line height, local overloads, half-space
  occupancy and more. Partial camera views are the norm in 360 data, so the description says
  what the camera cannot see instead of guessing.
- **Hybrid retrieval.** Dense search, BM25 and trigger-condition matching, fused with
  Reciprocal Rank Fusion and reranked by a cross-encoder. Scope gating keeps set-piece concepts
  out of open play and in-possession advice out of defensive questions.
- **Quantitative check.** A Spearman pitch-control model scores each candidate action by how
  much space it wins or loses. An Example Library of about 9,000 real StatsBomb situations adds
  what players actually did in similar spots.
- **Validated reasoning.** A LangGraph agent re-retrieves when evidence is weak. It regenerates
  when the output is malformed, cites a concept that was not retrieved, or gives vague advice.
  Recommended actions must come from the engine's candidate shortlist.
- **Natural answers.** A narrator rewrites the validated advice as a coach would say it, natively
  in the user's language. The numbers go into a separate "why" section, and every number in the
  narration is checked against the evidence.
- **Dashboard.**
  - A chat per player, with recommendation arrows drawn on the pitch.
  - Real team names, and attacking direction inferred per team.
  - Shirt numbers inferred for players who touched the ball moments earlier.
  - A "what happens next?" projection: the ball and the receiver move, and the other visible
    players react with physically capped rules. This costs zero extra LLM tokens.

## How it works

```
StatsBomb 360 snapshot
  └─► MatchState                 one Pydantic schema for every data source
       └─► encode_situation      features + a text description of the scene
            └─► render_pitch
                 └─► retrieve_tactics
                      │  weak evidence? ─► expand_query ─► back to retrieve_tactics
                      ▼
                     counterfactual_analysis     pitch control per candidate action
                      └─► retrieve_examples      similar real StatsBomb situations
                           └─► assemble_evidence
                                └─► generate_reasoning
                                     └─► validate_output
                                          │  fails? ─► retry ─► back to generate_reasoning (bounded)
                                          ▼
                                         narrate_advice  ─►  TacticalAdvice
                                                             (answer · rationale · citations · arrows)
```

Retrieval inside `retrieve_tactics`:

```
question + number-free tactical phrases from the situation
  → scope gating   (set pieces only in set-piece phases, possession-aware, named techniques only)
  → dense (MiniLM, exact cosine) + BM25 + trigger conditions
  → Reciprocal Rank Fusion → cross-encoder rerank (blended 50/50 with the fusion score)
  → top-k concepts
```

## Tech stack

**In use (V1)**

| Layer | Technology |
|---|---|
| Language and schema | Python 3.11+, Pydantic v2, pydantic-settings |
| Agent orchestration | LangGraph (conditional re-retrieve and regenerate loops) |
| LLMs | Gemini (`google-genai`), Groq (`groq`), Cerebras, Mistral and OpenRouter through the OpenAI-compatible API (`openai`); optional Anthropic |
| Vector store | ChromaDB (persistence) + exact cosine ranking in NumPy |
| Embeddings and reranking | sentence-transformers: `all-MiniLM-L6-v2` (dense), `cross-encoder/ms-marco-MiniLM-L-6-v2` (rerank) |
| Lexical search | rank-bm25 |
| Football data | statsbombpy + a local clone of StatsBomb Open Data (360 freeze frames, events, lineups) |
| Analytics | NumPy, SciPy (Hungarian assignment for shirt numbers), pandas; Spearman pitch-control model |
| Knowledge-base ingestion | youtube-transcript-api, yt-dlp (channel listings, no API key), httpx |
| Dashboard and visuals | Streamlit, matplotlib |
| CLI, tests, CI | Typer, Rich, pytest, GitHub Actions |

**Planned (V2/V3, declared in `pyproject.toml` but not used yet)**

| Layer | Technology |
|---|---|
| Live state and history | Redis, PostgreSQL (SQLAlchemy, asyncpg, Alembic) |
| Real-time API | FastAPI, WebSocket |
| Computer vision (V3) | YOLO detection, ByteTrack tracking, pitch homography |

## Evaluation

**Retrieval**: a golden set of 28 hand-labelled questions (`matchmind/evaluation/golden/retrieval_golden.yaml`;
27 are scored, one only checks for set-piece leaks), relevant = 2, acceptable = 1:

| Strategy | KB size | hit@3 | nDCG@5 | MRR | set-piece leak |
|---|---|---|---|---|---|
| dense only (pre-V1 baseline) | ~100 | 0.741 | 0.518 | 0.674 | 0.286 |
| hybrid + rerank (2026-09-28) | ~100 | 0.889 | 0.692 | 0.883 | 0 |
| hybrid + rerank, exact dense (2026-10-08) | 331 | 0.926 | 0.696 | 0.901 | 0 |
| **+ cross-phase scope fix (2026-10-14, current)** | **333** | **0.926** | **0.700** | **0.901** | **0** |

V1 targets are hit@3 ≥ 0.80, nDCG@5 ≥ 0.70, zero must-not hits and zero set-piece leaks
(current nDCG@5 is 0.6997, accepted as meeting the target). CI fails below hit@3 0.80 or nDCG@5 0.65.

The golden set has been used for tuning, so a second **held-out set** of 21 questions written by the
project owner (`retrieval_heldout.yaml`, never tuned on) keeps the numbers honest: hit@3 0.737,
nDCG@5 0.501, Recall@5 0.461. Long, multi-part questions remain the main weakness. Concepts already
carry LLM-written example questions (doc2query); turning them on (`RETRIEVAL_BM25_QUESTIONS=true`)
lifts held-out nDCG@5 to 0.538 but costs 0.013 on golden, so it is off by default.
`bench_retrieval` also reports KB coverage and whether scope gating wrongly blocks labelled concepts.

**End to end** (25 StatsBomb cases, 2026-10-15, 333-concept KB, `qwen/qwen3.8-27b`):

| Metric | Value |
|---|---|
| Citation validity | 1.00 |
| Logical consistency | 1.00 |
| Groundedness | 1.00 |
| Concept faithfulness (LLM judge; weakest metric) | 0.59 |
| Actionability (1–5, LLM rubric) | 4.12 |
| Agreement with the real player's action (reference only) | 0.60 |

```bash
python -m matchmind.scripts.bench_retrieval                 # retrieval benchmark on the golden set
python -m matchmind.scripts.run_evaluation --from-cache     # end-to-end evaluation (uses LLM tokens)
pytest matchmind/tests                                       # unit + integration tests
```

## Getting started

Requirements: Python 3.11+, and a local clone of [statsbomb/open-data](https://github.com/statsbomb/open-data)
next to this repo (`../open-data`, override with `STATSBOMB_OPEN_DATA_DIR`).

```bash
# 1. Install
pip install poetry && poetry install          # or: conda create -n matchmind python=3.11, then pip install -e .

#    then activate it (poetry shell / conda activate matchmind) before the commands below

# 2. Configure: add at least one LLM key (GROQ_API_KEY, CEREBRAS_API_KEY, GEMINI_API_KEY or MISTRAL_API_KEY)
cp .env.example .env
#    optional failover order, e.g. LLM_FALLBACK_CHAIN=groq:qwen/qwen3.8-27b,cerebras:qwen-3.8-27b,gemini:gemini-3.6-flash

# 3. Build the vector index from the concepts in the repo
python -m matchmind.scripts.build_kb --rebuild

# 4. Optional: test cases and the Example Library from the local StatsBomb clone
python -m matchmind.scripts.download_data --local
python -m matchmind.scripts.build_example_library

# 5. Run
streamlit run matchmind/dashboard/app.py                        # dashboard (loads matches via statsbombpy: needs internet)
python -m matchmind.scripts.demo --match-id 3788741 --event-idx 150   # one analysis in the terminal
python -m matchmind.scripts.demo --dual                         # attacking and defending side together
```

## Growing the knowledge base

Paste links into `matchmind/knowledge_base/sources/sources.txt`, one per line:

```
https://youtu.be/XXXXXXXXXXX                                   # one video
https://www.youtube.com/@thetacticsguy | max=10 | sort=popular # a channel's 10 most-viewed videos
```

```bash
python -m matchmind.scripts.ingest_sources --dry-run   # preview, no LLM calls
python -m matchmind.scripts.ingest_sources             # transcribe → extract concepts → lint → rebuild index
python -m matchmind.scripts.kb_dedup                   # list near-duplicate clusters (read-only)
python -m matchmind.scripts.kb_dedup --archive <id>    # move one aside (restorable with --restore)
python -m matchmind.scripts.kb_lint                    # schema and consistency checks
```

- **Safe to re-run.** Ingestion skips videos it has already processed.
- **Failures are retried.** A video whose LLM calls fail is retried on the next run, with nothing half-written.
- **Duplicates are skipped.** A new concept that is ≥ 0.90 cosine-similar to an existing one is not added.

Concepts are LLM paraphrases with a source link and timestamp; transcripts are never stored.
See [matchmind/knowledge_base/README.md](matchmind/knowledge_base/README.md) for the concept schema.

## Project structure

```
matchmind/
├── schema/            MatchState, TacticalConcept, TacticalAdvice (Pydantic v2)
├── data_sources/      StatsBomb (statsbombpy + local open-data), Metrica, jersey inference
├── situation_engine/  features, orientation, candidate actions, projection, team shift
├── pitch_control/     Spearman model + counterfactual analyzer
├── retrieval/         hybrid retriever: scope, BM25, fusion, rerank
├── knowledge_base/    concepts/, vector store, ingestion, dedup
├── examples/          Example Library (kNN over real StatsBomb situations)
├── agent/             LangGraph graph and nodes
├── evaluation/        retrieval bench, end-to-end evaluator, golden set
├── visualization/     pitch rendering
├── dashboard/         Streamlit app
├── scripts/           command-line entry points
└── tests/             pytest suite (~330 tests)
```

The top-level `api/`, `database/`, `state_engine/`, `observability/` folders and
`docker-compose.yml` (Redis, Postgres) are placeholders for V2 and contain no code yet.

## Roadmap

- **V1: RAG over snapshots (done).** All targets met on 2026-10-15. Known weak spots for later:
  concept faithfulness (0.59) and long multi-part questions (held-out nDCG@5 0.50).
- **V2: live.** Replay or stream match events into an always-current `MatchState`
  (Redis for the current state, Postgres for history). Situation triggers decide when an
  analysis is worth an LLM call, within a token budget. Insights are pushed to the dashboard
  over WebSocket.
- **V3: computer vision.** Player and ball detection (YOLO), tracking (ByteTrack), team
  assignment and pitch homography feed the **same** `MatchState`, so the agent, retrieval and
  dashboard need no changes.

## Data and licences

- **StatsBomb Open Data**: free for non-commercial use with attribution. Used for snapshots,
  evaluation cases and the Example Library.
- **Wikipedia**: CC BY-SA. Concepts derived from Wikipedia keep their source link.
- **FIFA Training Centre and YouTube**: paraphrased concepts with source links and timestamps.
  No transcripts or article text are redistributed.
- **Pitch control**: adapted from Laurie Shaw's [LaurieOnTracking](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking) (Spearman, 2018).
