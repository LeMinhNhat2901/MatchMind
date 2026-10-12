# MatchMind

**A football tactical assistant grounded in real match data and a curated tactics knowledge base.**

Pick a moment from a real match, pick a player, and ask what they should do. MatchMind reads the
situation from player positions, retrieves the relevant tactical concepts, checks the options
with a pitch-control model, and answers in plain language, in any language. Every answer cites
the concepts it used, and a validator rejects answers that cite anything it did not retrieve.

| | |
|---|---|
| **Status** | **V1 (RAG over StatsBomb 360 snapshots): feature-complete, final evaluation pending.** V2 (live) and V3 (computer vision) are planned, not started. See [Roadmap](#roadmap). |
| **Data** | [StatsBomb Open Data](https://github.com/statsbomb/open-data) 360 freeze frames (non-commercial, with attribution) |
| **Knowledge base** | 331 tactical concepts from hand-written notes, FIFA Training Centre, Wikipedia and YouTube tactics channels |
| **LLMs** | Free-tier friendly: a configurable failover chain across Groq, Cerebras, Gemini, Mistral and OpenRouter (single-provider Gemini by default) |

---

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

## Evaluation

**Retrieval**: a golden set of 28 hand-labelled questions (`matchmind/evaluation/golden/retrieval_golden.yaml`;
27 are scored, one only checks for set-piece leaks), relevant = 2, acceptable = 1:

| Strategy | KB size | hit@3 | nDCG@5 | MRR | set-piece leak |
|---|---|---|---|---|---|
| dense only (pre-V1 baseline) | ~100 | 0.741 | 0.518 | 0.674 | 0.286 |
| hybrid + rerank (2026-09-28) | ~100 | 0.889 | 0.692 | 0.883 | 0 |
| **hybrid + rerank (2026-10-08, current)** | **331** | **0.926** | **0.696** | **0.901** | **0** |

V1 targets are hit@3 ≥ 0.80, nDCG@5 ≥ 0.70, zero must-not hits and zero set-piece leaks.
nDCG@5 is 0.004 short. CI fails below hit@3 0.80 or nDCG@5 0.65.

**End to end** (25 StatsBomb cases, last run 2026-09-28 on the ~100-concept KB, to be re-run on the current KB):

| Metric | Value |
|---|---|
| Citation validity | 1.00 |
| Logical consistency | 1.00 |
| Groundedness | 0.95 |
| Concept faithfulness | 0.64 |
| Actionability (1–5, LLM rubric) | 4.2 |
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

- **V1: RAG over snapshots (now).** Close the last 0.004 of nDCG@5, re-run the end-to-end
  evaluation on the 331-concept KB, merge to `main`.
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
