# Project Workflow and Technical Overview

## End-to-end flow

1. **Input** – User provides text, image (OCR), audio (ASR), or PDF. Low-confidence extraction triggers HITL: user can edit/confirm before continuing.
2. **Parser** – Raw text → structured problem (topic, question, constraints). If `needs_clarification`, HITL: user confirms or edits parsed problem.
3. **Router** – Classifies intent (e.g. solve, explain concept) and topic for RAG.
4. **Retrieve** – RAG returns top-k chunks from Chroma; memory returns similar past problems (from Chroma semantic search when available, else SQLite recent + topic).
5. **Solver** – Generates solution steps and final answer using retrieved context and similar problems.
6. **Verifier** – Checks answer (units, domain, arithmetic). Low confidence → HITL: Approve or Reject.
7. **Explainer** – Produces step-by-step explanation. Result and feedback are stored in memory.

## The 5 agents

| Agent     | Role |
|----------|------|
| **Parser**   | Extract topic, question, constraints; set `needs_clarification` when ambiguous. |
| **Router**   | Intent and topic for retrieval. |
| **Solver**   | Step-by-step solution and final answer using RAG + memory. |
| **Verifier** | Correctness, units, domain; set `needs_hitl` when unsure. |
| **Explainer**| Tutor-style explanation from solution and verification. |

## Tech stack

- **LLM:** OpenAI (Parser, Router, Solver, Verifier, Explainer); vision API for image→text when `OPENAI_API_KEY` is set.
- **RAG:** Chroma (`data/chroma`), OpenAI embeddings by default (`text-embedding-3-small`).
- **Memory:** Sessions and feedback are stored in **SQLite** (`data/memory.db`) and in **Chroma** (collection `math_mentor_memory`). When using Chroma Cloud, memory persists across restarts. Similar-problem retrieval uses embedding similarity over Chroma when available; otherwise falls back to SQLite recent + topic filter.
- **Input:** Text (direct), Image (EasyOCR or vision LLM), Audio (openai-whisper, WAV without ffmpeg), PDF (PyMuPDF + vision/OCR for image pages).
- **Pipeline:** LangGraph in `src/agents/graph.py` (state graph with conditional edges for HITL).

## Design choices

- **Single pipeline entry:** `run_pipeline()` in `graph.py`; app and scripts use this only.
- **RAG:** No hallucinated citations; only retrieved chunks are shown and passed to the Solver.
- **HITL:** At extraction, parser, and verifier; no model retraining, only human confirm/reject.
- **Memory:** Feedback (Correct/Incorrect) stored per session in SQLite and Chroma; Chroma enables cloud persistence and semantic similar-problem retrieval. Retrieval augments Solver context; no model retraining.
