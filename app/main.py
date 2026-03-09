"""Math Mentor Streamlit app. Run from repo root: streamlit run app/main.py"""
import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

# Inject Streamlit Cloud Secrets into env so config and agents see them
try:
    if hasattr(st, "secrets"):
        import os
        if st.secrets.get("OPENAI_API_KEY"):
            os.environ.setdefault("OPENAI_API_KEY", str(st.secrets["OPENAI_API_KEY"]))
        if st.secrets.get("CHROMA_API_KEY"):
            os.environ.setdefault("CHROMA_API_KEY", str(st.secrets["CHROMA_API_KEY"]))
        if st.secrets.get("CHROMA_TENANT"):
            os.environ.setdefault("CHROMA_TENANT", str(st.secrets["CHROMA_TENANT"]))
        if st.secrets.get("CHROMA_DATABASE"):
            os.environ.setdefault("CHROMA_DATABASE", str(st.secrets["CHROMA_DATABASE"]))
except Exception:
    pass

from src.config import OCR_CONFIDENCE_THRESHOLD, OPENAI_API_KEY
from src.input.text_parser import parse_text
from src.input.image_parser import parse_image
from src.input.audio_parser import parse_audio
from src.input.pdf_parser import parse_pdf
from src.rag.retriever import retrieve
from src.agents.graph import run_pipeline
from src.memory.store import store

st.set_page_config(page_title="Math Mentor", layout="wide")
st.title("Math Mentor")
st.caption("Multimodal math problem solver with RAG, agents, and HITL")

if not OPENAI_API_KEY or not OPENAI_API_KEY.strip().startswith("sk-"):
    st.error(
        "**OpenAI API key not set.** Solving will not work until you add it. "
        "**Local:** Put `OPENAI_API_KEY=sk-your-key` in the `.env` file in the project root and restart the app. "
        "**Streamlit Cloud:** Go to app Settings → Secrets and add: `OPENAI_API_KEY = \"sk-your-key\"` then save."
    )

# Session state
if "result" not in st.session_state:
    st.session_state.result = None
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""
if "hitl_stage" not in st.session_state:
    st.session_state.hitl_stage = None  # "extraction" | "parser" | "verifier"
if "parsed_override" not in st.session_state:
    st.session_state.parsed_override = None
if "skip_verify_hitl" not in st.session_state:
    st.session_state.skip_verify_hitl = False
if "show_incorrect_form" not in st.session_state:
    st.session_state.show_incorrect_form = False

# Sidebar: input mode
mode = st.sidebar.radio("Input mode", ["Text", "Image", "Audio", "PDF"], index=0)

def get_raw_text_from_ui():
    """Get raw text from current mode and optional extraction preview."""
    if mode == "Text":
        raw = st.session_state.get("text_input", "")
        return parse_text(raw)
    if mode == "Image":
        return st.session_state.get("extraction_result", None)
    if mode == "Audio":
        return st.session_state.get("audio_extraction_result", None)
    if mode == "PDF":
        return st.session_state.get("pdf_extraction_result", None)
    return None

# Main area
if mode == "Text":
    text_input = st.text_area("Enter your math problem", height=120, key="text_input")
    confirm_text = st.button("Solve")
    if confirm_text and text_input.strip():
        extraction = parse_text(text_input)
        st.session_state.raw_text = extraction["text"]
        st.session_state.hitl_stage = None
        def get_chunks(parsed):
            return retrieve(parsed)
        result = run_pipeline(st.session_state.raw_text, get_chunks)
        st.session_state.result = result
        st.rerun()

elif mode == "Image":
    img_file = st.file_uploader("Upload image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if img_file is not None:
        bytes_data = img_file.read()
        extraction = parse_image(bytes_data, confidence_threshold=OCR_CONFIDENCE_THRESHOLD)
        if "extraction_result" not in st.session_state or st.session_state.get("last_image_id") != id(img_file):
            st.session_state.extraction_result = extraction
            st.session_state.last_image_id = id(img_file)
        extraction = st.session_state.extraction_result
        st.session_state.raw_text = extraction.get("text", "")
        needs_hitl = extraction.get("needs_hitl", False)
        st.write("**Extracted text** (edit if needed):")
        edited = st.text_area("Extracted text", value=extraction.get("text", ""), height=100, key="image_edited_text", label_visibility="collapsed")
        if st.button("Confirm and Solve"):
            st.session_state.raw_text = edited or extraction.get("text", "")
            st.session_state.hitl_stage = None
            def get_chunks(parsed):
                return retrieve(parsed)
            result = run_pipeline(st.session_state.raw_text, get_chunks)
            st.session_state.result = result
            st.rerun()

elif mode == "Audio":
    audio_file = st.file_uploader("Upload audio (WAV/MP3)", type=["wav", "mp3", "m4a"])
    if audio_file is not None:
        bytes_data = audio_file.read()
        extraction = parse_audio(bytes_data, confidence_threshold=0.6, filename=getattr(audio_file, "name", None))
        if "audio_extraction_result" not in st.session_state or st.session_state.get("last_audio_id") != id(audio_file):
            st.session_state.audio_extraction_result = extraction
            st.session_state.last_audio_id = id(audio_file)
        extraction = st.session_state.audio_extraction_result
        st.session_state.raw_text = extraction.get("text", "")
        st.write("**Transcript** (edit if needed):")
        edited = st.text_area("Transcript", value=extraction.get("text", ""), height=100, key="audio_edited_text", label_visibility="collapsed")
        if st.button("Confirm and Solve"):
            st.session_state.raw_text = edited or extraction.get("text", "")
            st.session_state.hitl_stage = None
            def get_chunks(parsed):
                return retrieve(parsed)
            result = run_pipeline(st.session_state.raw_text, get_chunks)
            st.session_state.result = result
            st.rerun()

elif mode == "PDF":
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    if pdf_file is not None:
        bytes_data = pdf_file.read()
        extraction = parse_pdf(bytes_data, confidence_threshold=OCR_CONFIDENCE_THRESHOLD)
        if "pdf_extraction_result" not in st.session_state or st.session_state.get("last_pdf_id") != id(pdf_file):
            st.session_state.pdf_extraction_result = extraction
            st.session_state.last_pdf_id = id(pdf_file)
        extraction = st.session_state.pdf_extraction_result
        st.session_state.raw_text = extraction.get("text", "")
        st.write("**Extracted text** (edit if needed):")
        edited = st.text_area("PDF text", value=extraction.get("text", ""), height=150, key="pdf_edited_text", label_visibility="collapsed")
        if st.button("Confirm and Solve", key="pdf_confirm"):
            st.session_state.raw_text = edited or extraction.get("text", "")
            st.session_state.hitl_stage = None
            def get_chunks(parsed):
                return retrieve(parsed)
            result = run_pipeline(st.session_state.raw_text, get_chunks)
            st.session_state.result = result
            st.rerun()

# Handle HITL and display result
result = st.session_state.result
if result is not None:
    if result.get("hitl_required") == "parser":
        st.warning("The parser needs clarification. Please confirm or edit the parsed problem below.")
        parsed = result.get("parsed", {})
        st.json(parsed)
        if st.button("Confirm and continue"):
            st.session_state.parsed_override = parsed
            st.session_state.result = None
            def get_chunks(p):
                return retrieve(p)
            r = run_pipeline(st.session_state.raw_text, get_chunks, parsed_override=parsed)
            st.session_state.result = r
            st.session_state.parsed_override = None
            st.rerun()
    elif result.get("hitl_required") == "verifier":
        st.warning("Verifier has low confidence. Please approve or reject the solution.")
        st.write("**Solution:**")
        sol = result.get("solution", {})
        for i, step in enumerate(sol.get("steps", []), 1):
            st.write(f"{i}. {step}")
        st.write("**Final answer:**", sol.get("final_answer"))
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Approve (continue)"):
                def get_chunks(p):
                    return retrieve(p)
                r = run_pipeline(
                    st.session_state.raw_text,
                    get_chunks,
                    parsed_override=result.get("parsed"),
                    solution_override=result.get("solution"),
                    verification_override=result.get("verification"),
                )
                st.session_state.result = r
                st.rerun()
        with col2:
            if st.button("Reject (start over)"):
                st.session_state.result = None
                st.rerun()
    else:
        # Full result: trace, context, answer, explanation, confidence, feedback
        with st.expander("Agent trace", expanded=False):
            for t in result.get("trace", []):
                st.write(f"**{t.get('step', '')}**", t.get("output", ""))
        retrieved = result.get("retrieved", [])
        st.subheader("Retrieved context")
        if retrieved:
            for i, c in enumerate(retrieved, 1):
                st.caption(f"[{c.get('source', 'doc')}]")
                st.text(c.get("text", "")[:500] + ("..." if len(c.get("text", "")) > 500 else ""))
        else:
            st.caption("No chunks retrieved (empty knowledge base or no match).")
        sol = result.get("solution", {})
        ver = result.get("verification", {})
        st.subheader("Final answer")
        st.write(sol.get("final_answer", "N/A"))
        conf = ver.get("confidence", 0)
        if conf >= 0.75:
            st.success(f"Confidence: High ({conf:.0%})")
        elif conf >= 0.5:
            st.info(f"Confidence: Medium ({conf:.0%})")
        else:
            st.warning(f"Confidence: Low ({conf:.0%})")
        st.subheader("Explanation")
        st.markdown(result.get("explanation", ""))
        # Re-check: re-run pipeline with same input (user explicitly requests re-verification)
        if st.button("Re-check solution", key="recheck_btn"):
            def get_chunks(p):
                return retrieve(p)
            r = run_pipeline(st.session_state.raw_text, get_chunks)
            st.session_state.result = r
            st.rerun()
        # Feedback
        st.divider()
        st.write("Was this solution correct?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Correct"):
                store(
                    st.session_state.raw_text,
                    result.get("parsed", {}),
                    result.get("retrieved", []),
                    result.get("solution", {}),
                    result.get("verification", {}),
                    user_feedback="correct",
                )
                st.success("Thanks! Feedback saved.")
                st.rerun()
        with col2:
            if st.button("Incorrect"):
                st.session_state.show_incorrect_form = True
                st.rerun()
        if st.session_state.get("show_incorrect_form"):
            comment = st.text_input("Optional comment (what was wrong?)", key="feedback_comment_incorrect")
            if st.button("Submit feedback", key="submit_incorrect"):
                store(
                    st.session_state.raw_text,
                    result.get("parsed", {}),
                    result.get("retrieved", []),
                    result.get("solution", {}),
                    result.get("verification", {}),
                    user_feedback="incorrect",
                    feedback_comment=comment,
                )
                st.session_state.show_incorrect_form = False
                st.success("Feedback saved. We'll improve from this.")
                st.rerun()
