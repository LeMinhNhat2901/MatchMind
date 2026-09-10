# MatchMind 🧠⚽

**Real-Time Multi-Agent Football Tactical Intelligence**

A production-grade AI system that converts player tracking data and live match events into structured tactical recommendations using a multi-agent reasoning pipeline with RAG, quantitative pitch control analysis, and counterfactual reasoning.

---

## Architecture

```
FOOTBALL DATA SOURCES
       │
┌──────┼──────┐
▼      ▼      ▼
StatsBomb  Live API  Match Video
Metrica    WebSocket (Phase 3)
       │
       ▼
Unified Match Schema (Pydantic)
       │
       ▼
Match State Engine (Redis + PostgreSQL)
       │
       ▼
Situation Representation
├── Spatial Engine     (distances, angles, free space)
├── Team Structure     (compactness, defensive line)
└── Game Context       (score, phase, possession)
       │
       ▼
Evidence Assembly
├── Tactical KB (RAG)  — static football knowledge
├── Computed Stats     — quantitative + pitch control
└── Match Context      — dynamic match state
       │
       ▼
Tactical Agent (LangGraph)
├── retrieve → [low confidence?] → re-retrieve
├── generate → [validation fail?] → regenerate
└── validate (anti-hallucination)
       │
       ▼
Counterfactual Engine (Pitch Control Model)
"Action A opens +18% control area vs Action B"
       │
       ▼
Structured Output (Pydantic)
├── Dashboard (Streamlit)
└── Evaluation (pytest CI)
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install poetry
cd matchmind
poetry install
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Build knowledge base
```bash
poetry run python -m matchmind.scripts.build_kb
```

### 4. Run demo
```bash
poetry run python -m matchmind.scripts.demo --player 14
```

### 5. Launch dashboard
```bash
poetry run streamlit run matchmind/dashboard/app.py
```

### 6. Run evaluation
```bash
poetry run python -m matchmind.scripts.run_evaluation --n-cases 25
```

---

## Project Structure

```
matchmind/
├── schema/                 # Unified data contract (Pydantic)
├── data_sources/           # StatsBomb, Metrica, Live adapters
├── situation_engine/       # Feature computation (pure Python)
├── pitch_control/          # Spearman pitch control + counterfactuals
├── knowledge_base/         # 30+ tactical concepts + ChromaDB
├── agent/                  # LangGraph multi-node agent
├── visualization/          # Pitch renderer (matplotlib)
├── evaluation/             # 4-dim evaluation framework
├── api/                    # FastAPI + WebSocket
├── dashboard/              # Streamlit UI
└── scripts/                # CLI entry points
```

---

## Evaluation (4 Dimensions)

| Dimension | What it measures | Metric |
|-----------|-----------------|--------|
| **Agreement** | Agent vs actual player action | `agreement_rate` (25 StatsBomb cases) |
| **Groundedness** | Citations valid, no hallucination | `citation_validity_rate`, `logical_consistency` |
| **Actionability** | Advice specific enough to act on | `actionability_score` (1–5 LLM rubric) |
| **Consistency** | Stable output + aligned with pitch control | `consistency_rate`, `quantitative_alignment` |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Schema | Pydantic v2 |
| Agent | LangGraph + LangChain |
| LLM | Claude (Anthropic) / GPT-4o |
| Vector DB | ChromaDB |
| Embeddings | sentence-transformers |
| Football data | StatsBombPy, kloppy |
| Pitch control | Spearman model (adapted from LaurieOnTracking) |
| API | FastAPI + WebSocket |
| Dashboard | Streamlit + Plotly |
| State store | Redis (current) + PostgreSQL (history) |
| Containers | Docker Compose |

---

## Milestones

- **MVP (Phase 1):** StatsBomb/Metrica → Schema → Situation Engine → RAG → Agent → Evaluation → Dashboard
- **V2 (Phase 2):** Live event stream → Redis state engine → WebSocket real-time
- **V3 (Phase 3):** YOLO + ByteTrack + Homography → Video CV → Existing agent (no agent changes needed)

---

## CV Bullets

- Designed a unified match-state schema (Pydantic) decoupling tactical reasoning from tracking/CV data sources
- Built a computational situation engine computing 11+ spatial/tactical features from raw player coordinates
- Integrated Spearman pitch control model as quantitative backbone for counterfactual reasoning
- Implemented LangGraph agent with real conditional branching (re-retrieve + regenerate loops)
- Developed 4-dimensional evaluation framework: agreement, groundedness, actionability, consistency
- Extended pipeline with video-based player tracking via YOLO, ByteTrack, and homography (Phase 3)
