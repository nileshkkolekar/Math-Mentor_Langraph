# Math Mentor – Project Workflow & Technical Overview

This document describes the **end-to-end workflow**, **agents**, **technical stack**, and **design decisions** so you can understand how the whole system works.

---

## 1. End-to-End Workflow

### High-level flow

```
User Input (Text / Image / Audio)
        │
        ▼
┌───────────────────┐
│ Multimodal Parse  │  OCR (image) / ASR (audio) / text as-is → raw text
│ + Extraction UI  │  If low confidence → HITL: user edits/confirms
└─────────┬─────────┘
          │ raw text
          ▼
┌───────────────────┐
│ 1. Parser Agent   │  Raw text → structured { problem_text, topic, variables, constraints, needs_clarification }
│                   │  If needs_clarification → HITL: user confirms/edits
└─────────┬─────────┘
          │ parsed
          ▼
┌───────────────────┐
│ 2. Router Agent   │  Classify topic + subtype + strategy hint
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ 3. RAG + Memory   │  Vector search (Chroma) → top-k KB chunks
│                   │  SQLite memory → similar past problems (by topic)
└─────────┬─────────┘
          │ retrieved chunks + similar problems
          ▼
┌───────────────────┐
│ 4. Solver Agent   │  LLM + RAG context + similar solutions → steps + final answer
└─────────┬─────────┘
          │ solution
          ▼
┌───────────────────┐
│ 5. Verifier Agent │  Correct? Units? Domain? → { correct, confidence, issues }
│                   │  If confidence < threshold → HITL: Approve or Reject
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ 6. Explainer Agent│  Step-by-step student-friendly explanation (markdown)
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ UI: Answer +      │  Show trace, retrieved context, confidence, explanation
│ Explanation +     │  Feedback: Correct / Incorrect + comment → stored in memory
│ Feedback          │
└───────────────────┘
```

### Step-by-step (what happens when you click “Solve”)

| Step | Component | Input | Output | Notes |
|------|-----------|--------|--------|--------|
| 0 | **Input layer** | Image bytes / audio bytes / text | `raw_text` (string) | Image: EasyOCR. Audio: Whisper. Text: as-is. UI shows extraction for edit before Solve. |
| 1 | **Parser Agent** | `raw_text` | `parsed`: problem_text, topic, variables, constraints, needs_clarification | LLM (OpenAI). If needs_clarification → stop, show parsed JSON, HITL confirm. |
| 2 | **Router Agent** | `parsed` | topic, subtype, strategy_hint | LLM classifies and gives a one-line solving hint. |
| 3 | **RAG** | `parsed` (problem_text + topic) | List of chunks `[{ text, source }]` | Embed query → Chroma top-k. Only these chunks are shown and used; no hallucinated citations. |
| 3b | **Memory** | `parsed` (topic) | List of similar past sessions | Recent sessions from SQLite (same topic). Passed to Solver for pattern reuse. |
| 4 | **Solver Agent** | parsed, route_info, chunks, similar_problems | steps[], final_answer | LLM with RAG + similar solutions in prompt. Optional: sympy tool for numeric/symbolic math. |
| 5 | **Verifier Agent** | parsed, solution | correct, confidence, issues, needs_hitl | LLM checks correctness/units/domain. If confidence < threshold → stop, HITL Approve/Reject. |
| 6 | **Explainer Agent** | parsed, solution, verification | Markdown explanation | LLM turns solution into a clear, step-by-step tutor-style explanation. |
| 7 | **UI + Memory** | Full result + user feedback | — | User sees answer, explanation, confidence; can click Correct / Incorrect. Feedback stored in SQLite for future similar-problem retrieval. |

---

## 2. The Five Agents

The system uses **exactly 5 agents** (all LLM-based unless noted).

| # | Agent | File | Role | Why separate |
|---|--------|------|------|------------------|
| 1 | **Parser Agent** | `src/agents/parser_agent.py` | Converts raw text (OCR/ASR/typed) into a **structured** math problem: `problem_text`, `topic`, `variables`, `constraints`, `needs_clarification`. | Single responsibility: normalize and structure. Enables downstream routing and RAG to work on clean, typed fields. HITL when ambiguous. |
| 2 | **Intent Router Agent** | `src/agents/router_agent.py` | Classifies **topic** (algebra / probability / calculus / linear_algebra), **subtype**, and a short **strategy_hint** for the Solver. | Decouples “what kind of problem” from “how to solve.” Lets Solver prompt be topic-aware without one giant prompt. |
| 3 | **Solver Agent** | `src/agents/solver_agent.py` | Produces **solution steps** and **final answer** using RAG chunks + similar past problems. Can use tools (e.g. sympy) for exact math. | Central reasoning step. Keeps RAG and memory in one place; Verifier and Explainer stay independent of how the solution was found. |
| 4 | **Verifier / Critic Agent** | `src/agents/verifier_agent.py` | Checks **correctness**, **units**, **domain**, **edge cases**. Returns `correct`, `confidence`, `issues`. Triggers HITL when confidence is low. | Catches errors before showing the user. Separate from Solver so the model doesn’t “mark its own homework”; explicit confidence enables HITL. |
| 5 | **Explainer / Tutor Agent** | `src/agents/explainer_agent.py` | Turns the (verified) solution into a **step-by-step, student-friendly** explanation in markdown. | Pedagogy is different from solving. One model for solving, one for explaining keeps outputs clear and consistent. |

**Orchestrator** (`src/agents/orchestrator.py`) runs them in order: Parser → Router → Retrieve (RAG + memory) → Solver → Verifier → Explainer, and handles HITL returns (parser clarification, verifier approve/reject).

---

## 3. Technical Stack

| Layer | Technology | Purpose |
|-------|------------|--------|
| **UI** | Streamlit | Single-page app: mode selector, extraction preview, agent trace, retrieved context, answer, explanation, confidence, feedback buttons. |
| **Language** | Python 3.11+ | Entire backend and app. |
| **LLM** | OpenAI API (e.g. gpt-4o-mini) | All 5 agents use the same API; model name in config. |
| **Image → text** | EasyOCR | OCR on JPG/PNG. Returns text + confidence; no API key. |
| **Audio → text** | OpenAI Whisper (openai-whisper) | ASR for WAV/MP3. Local model “base”; math-phrase normalization in code. |
| **RAG** | sentence-transformers + Chroma | Embeddings: `all-MiniLM-L6-v2`. Vector store: Chroma (persistent on disk). Chunking in `src/rag/chunker.py`. |
| **Knowledge base** | Markdown files in `knowledge_base/` | 10+ docs: algebra, probability, calculus, linear algebra, solution templates, common mistakes. Chunked and indexed by `scripts/build_rag.py`. |
| **Memory** | SQLite | Table `sessions`: original_input, parsed_question, retrieved_context, solution, verifier_outcome, user_feedback, feedback_comment. Used for similar-problem retrieval (no vector search over history in minimal version). |
| **Math tools** | SymPy | Safe symbolic/numeric math in `src/tools/calculator.py` (evaluate, differentiate, limit). No arbitrary code execution. |
| **Config** | python-dotenv + `.env` | API keys, model names, paths, HITL thresholds (OCR confidence, verifier confidence). |

---

## 4. Design Decisions: “Why This Instead of That?”

| Decision | Choice | Alternative | Reason |
|----------|--------|-------------|--------|
| **UI framework** | Streamlit | React / FastAPI + frontend | Streamlit: quick to build, single Python codebase, good for demos and deployment (e.g. Streamlit Cloud). No separate frontend/backend wiring. |
| **OCR** | EasyOCR | Tesseract, PaddleOCR, cloud OCR | EasyOCR: Python-native, no system binaries, works out of the box on Windows. PaddleOCR is heavier; Tesseract needs install; cloud adds cost and latency. |
| **ASR** | OpenAI Whisper (openai-whisper) | Cloud ASR (e.g. Google, AWS) | Whisper: strong open-source model, runs locally, no per-request cost. Use “base” for speed; “small”/“medium” for better accuracy. |
| **Vector store** | Chroma | FAISS, Pinecone, Weaviate | Chroma: simple, persistent, good for small-to-medium KB. No extra service; same process. FAISS is in-memory unless you persist yourself. |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | OpenAI embeddings | Local, no per-request cost, good for 10–30 docs. OpenAI embeddings are simpler to use but add API cost and latency. |
| **Agents** | 5 separate agents | 1–2 monolithic prompts | Separation: clearer prompts, easier to debug and improve each step (parse vs route vs solve vs verify vs explain). HITL can be attached to Parser and Verifier only. |
| **Memory** | SQLite + “recent by topic” | Vector search over past problems | SQLite: no extra infra, stores full sessions and feedback. Similar-problem reuse by topic + recency is enough for the assignment; vector search over history would be a natural upgrade. |
| **HITL** | In-UI (confirm/edit/approve/reject) | Separate review dashboard | In-UI: student stays in one place; no extra app. Corrections and approvals are stored as learning signals in the same `sessions` table. |
| **Calculator** | SymPy in a dedicated module | LLM-generated code / eval() | SymPy: safe (no arbitrary execution), good for limits/derivatives/algebra. Avoids security and reliability issues of raw code execution. |

---

## 5. Data Flow (Simplified)

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Image /   │     │  Parser     │     │   Router    │
│   Audio /   │ ──► │  Agent      │ ──► │   Agent     │
│   Text      │     │  (LLM)      │     │   (LLM)     │
└─────────────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       │                   │ parsed             │ route_info
       │                   ▼                   ▼
       │            ┌─────────────┐     ┌─────────────┐
       │            │  RAG        │     │  Memory     │
       │            │  (Chroma)   │     │  (SQLite)   │
       │            └──────┬──────┘     └──────┬──────┘
       │                   │ chunks            │ similar
       │                   └────────┬──────────┘
       │                            ▼
       │                     ┌─────────────┐
       │                     │  Solver     │
       │                     │  Agent(LLM) │
       │                     └──────┬──────┘
       │                            │ solution
       │                            ▼
       │                     ┌─────────────┐
       │                     │  Verifier   │
       │                     │  Agent(LLM) │
       │                     └──────┬──────┘
       │                            │ verification
       │                            ▼
       │                     ┌─────────────┐
       │                     │  Explainer  │
       │                     │  Agent(LLM) │
       │                     └──────┬──────┘
       │                            │ explanation
       │                            ▼
       └────────────────────► ┌─────────────┐
                              │  UI +       │
                              │  Feedback   │ ──► store() for Correct/Incorrect
                              └─────────────┘
```

---

## 6. HITL (Human-in-the-Loop) Points

| Trigger | Where | What the user sees | Actions |
|--------|--------|---------------------|--------|
| **Low OCR/ASR confidence** | Input layer (before pipeline) | Extracted text or transcript with an edit box | User edits if wrong, then clicks “Confirm and Solve”. |
| **Parser: needs_clarification** | After Parser Agent | Parsed JSON (problem_text, topic, etc.) | User clicks “Confirm and continue” (or could edit and re-run). |
| **Verifier: low confidence** | After Verifier Agent | Solution steps + final answer | **Approve (continue)** → keep solution, run Explainer only; **Reject** → clear result, start over. |
| **Re-check** | After full result | Button “Re-check solution” | Re-runs the full pipeline (same raw text) to get a fresh solution and verification. |

Approved or corrected outcomes are stored in the **memory** (sessions table) and used for **similar-problem retrieval** in future runs (pattern reuse, no model retraining).

---

## 7. File Map (Where Things Live)

| Concern | Files |
|---------|--------|
| **Config** | `src/config.py`, `.env.example`, `.env` |
| **Input** | `src/input/text_parser.py`, `image_parser.py`, `audio_parser.py` |
| **Agents** | `src/agents/parser_agent.py`, `router_agent.py`, `solver_agent.py`, `verifier_agent.py`, `explainer_agent.py`, `orchestrator.py` |
| **Tools** | `src/tools/calculator.py` (SymPy) |
| **RAG** | `src/rag/chunker.py`, `embedder.py`, `vector_store.py`, `retriever.py` |
| **Knowledge base** | `knowledge_base/*.md` |
| **RAG build** | `scripts/build_rag.py` |
| **Memory** | `src/memory/store.py`, `retriever.py` |
| **HITL** | Handled in `orchestrator.py` and `app/main.py` (no separate HITL package logic) |
| **UI** | `app/main.py` |

---

## 8. Summary

- **Workflow:** Multimodal input → (optional HITL on extraction) → Parser → (optional HITL on ambiguity) → Router → RAG + Memory → Solver → Verifier → (optional HITL on low confidence) → Explainer → UI and feedback → Memory.
- **Agents:** 5 (Parser, Router, Solver, Verifier, Explainer), all driven by the same LLM API, with a single orchestrator.
- **Stack:** Python, Streamlit, OpenAI API, EasyOCR, Whisper, sentence-transformers, Chroma, SQLite, SymPy.
- **Why:** Streamlit for speed and deployability; local OCR/ASR and embeddings to limit cost; separate agents for clarity and HITL; Chroma and SQLite to avoid extra services; SymPy for safe math; memory for pattern reuse without retraining.

This gives you the full picture of the project workflow and how each part fits together end to end.
