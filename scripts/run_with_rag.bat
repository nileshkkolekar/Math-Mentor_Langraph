@echo off
REM Rebuild RAG index if missing, then start Streamlit. Windows.
python scripts/build_rag.py 2>nul
if not defined PORT set PORT=8501
streamlit run app/main.py --server.port %PORT%
