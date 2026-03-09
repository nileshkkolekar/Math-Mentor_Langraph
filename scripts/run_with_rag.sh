#!/usr/bin/env bash
# Rebuild RAG index if missing, then start Streamlit. Use as start command on cloud (e.g. ./scripts/run_with_rag.sh).
set -e
python scripts/build_rag.py 2>/dev/null || true
exec streamlit run app/main.py --server.port "${PORT:-8501}"
