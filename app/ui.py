"""Streamlit front end. It only talks to the FastAPI backend over HTTP."""
from __future__ import annotations

import html
import json
import os

import requests
import streamlit as st

API = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/") + "/api/v1"
FILE_TYPES = ["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"]
SEARCH_METHODS = {
    "hybrid": "Hybrid (keywords + language model + embeddings)",
    "tfidf": "Keywords only (TF-IDF)",
    "lm": "Language model only",
    "semantic": "Embeddings only",
}

st.set_page_config(page_title="University Knowledge Assistant", page_icon="📄", layout="wide")

if "conversation" not in st.session_state:
    st.session_state.conversation = []


class ApiError(Exception):
    pass


def call(method: str, path: str, timeout: int = 60, **kwargs):
    try:
        response = requests.request(method, f"{API}{path}", timeout=timeout, **kwargs)
    except requests.exceptions.ConnectionError as exc:
        raise ApiError("Can't reach the backend. Start it with: python run.py") from exc
    except requests.exceptions.Timeout as exc:
        raise ApiError("The request timed out.") from exc
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ApiError(str(detail))
    return response.json()


def safe(method: str, path: str, default=None, **kwargs):
    try:
        return call(method, path, **kwargs)
    except ApiError:
        return default


def highlighted(text: str, sentences: list[str]) -> str:
    """The passage as HTML with the sentences used for the answer marked."""
    out = html.escape(text)
    for sentence in sorted(set(sentences), key=len, reverse=True):
        escaped = html.escape(sentence)
        out = out.replace(escaped, f"<mark>{escaped}</mark>", 1)
    return f"<div style='white-space:pre-wrap;font-size:0.9rem'>{out}</div>"


# ---------------------------------------------------------------- sidebar

health = safe("GET", "/health", timeout=5)
documents = safe("GET", "/documents", default=[]) if health else []

with st.sidebar:
    st.title("Knowledge base")
    if health is None:
        st.error("The backend isn't running. Start it with `python run.py`.")
    else:
        st.metric("Indexed documents", len(documents))
        st.caption(
            f"Answers: {'Gemini' if health['llm_configured'] else 'sentences taken from the documents (no API key)'}\n\n"
            f"Embedding search: {'on' if health['semantic_search'] else 'off (keyword search only)'}\n\n"
            f"OCR: {health['ocr_engine'] or 'not installed'}"
        )
        if st.button("Scan documents/ folder", use_container_width=True):
            with st.spinner("Reading new files..."):
                try:
                    result = call("POST", "/reindex", timeout=900)
                    st.success(f"{result['indexed_documents']} indexed, {result['newly_indexed']} new.")
                except ApiError as exc:
                    st.error(str(exc))

tab_ask, tab_docs = st.tabs(["Ask", "Documents"])

# ---------------------------------------------------------------- ask

with tab_ask:
    st.header("Ask your documents")
    st.write("Answers come only from the files you've added. If the answer isn't in them, it says so.")

    question = st.text_area("Question", height=90, placeholder="What is the minimum attendance needed for the exam?")
    left, middle, right = st.columns([2, 1, 1])
    mode = left.selectbox("Search method", list(SEARCH_METHODS), format_func=SEARCH_METHODS.get)
    middle.write("")
    ask = middle.button("Ask", type="primary", use_container_width=True)
    right.write("")
    if right.button("New conversation", use_container_width=True):
        st.session_state.conversation = []
        st.rerun()

    if ask:
        if not question.strip():
            st.warning("Type a question first.")
        else:
            try:
                with st.spinner("Searching the documents..."):
                    result = call(
                        "POST", "/ask", timeout=300,
                        data={"question": question, "mode": mode, "history": json.dumps(st.session_state.conversation)},
                    )
            except ApiError as exc:
                st.error(str(exc))
            else:
                st.session_state.conversation.append({"question": question, "answer": result["answer"]})

                if result["evidence_sufficient"]:
                    st.subheader("Answer")
                    st.write(result["answer"])
                else:
                    st.warning(result["answer"])
                    st.caption(f"Why: {result['reason']}")
                    if result["suggestions"]:
                        st.write("The files do cover topics such as: " + "; ".join(result["suggestions"]))

                a, b, c, d = st.columns(4)
                a.metric("Confidence", result["confidence"])
                b.metric("Evidence strength", f"{result['retrieval_score']:.2f}")
                c.metric("Found in documents", "Yes" if result["evidence_sufficient"] else "No")
                d.metric("Question type", result["analysis"].get("question_type", "").replace("_", "/"))

                likely = result["likely_documents"]
                if likely:
                    st.caption(
                        "Most likely source (Naive Bayes on the question): "
                        + ", ".join(f"{x['document']} ({x['probability']:.0%})" for x in likely[:2])
                    )

                if result["citations"]:
                    st.subheader("Sources")
                    st.caption("The sentences the answer is based on are highlighted.")
                    for number, cite in enumerate(result["citations"], start=1):
                        with st.expander(f"[{number}] {cite['document_name']}, page {cite['page']}, {cite['section']}"):
                            st.caption(f"{cite['document_type']} | match score {cite['score']:.2f}")
                            st.markdown(highlighted(cite["text"], cite["highlights"]), unsafe_allow_html=True)

                with st.expander("How this was decided"):
                    st.write("Words searched for: " + ", ".join(result["analysis"].get("terms", [])))
                    st.json(result["signals"])

    if len(st.session_state.conversation) > 1:
        with st.expander(f"Earlier questions ({len(st.session_state.conversation)})"):
            for number, turn in enumerate(st.session_state.conversation, start=1):
                st.markdown(f"**{number}. {turn['question']}**")
                st.write(turn["answer"])

# ---------------------------------------------------------------- documents

with tab_docs:
    st.header("Documents")
    st.write(
        "Files in the `documents/` folder are read when the server starts. "
        "You can also upload more here. Scanned PDFs and photos of text are read with OCR."
    )

    uploads = st.file_uploader("Add files", type=FILE_TYPES, accept_multiple_files=True)
    if st.button("Index uploaded files", disabled=not uploads):
        files = [("files", (u.name, u.getvalue(), "application/octet-stream")) for u in uploads]
        with st.spinner("Reading the files (OCR can take a minute per page)..."):
            try:
                result = call("POST", "/index", timeout=1800, files=files)
                if result["newly_indexed"]:
                    st.success("Added: " + ", ".join(result["newly_indexed"]))
                if result["reused"]:
                    st.info("Already indexed: " + ", ".join(result["reused"]))
                st.rerun()
            except ApiError as exc:
                st.error(str(exc))

    if documents:
        rows = [
            {
                "File": d["document_name"],
                "Type": d["document_type"],
                "Pages": d.get("page_count", 0),
                "Read with OCR": len(d.get("ocr_pages", [])),
                "Chunks": d["chunk_count"],
            }
            for d in documents
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        if any(d.get("ocr_pages") for d in documents):
            st.caption("OCR text can contain misread characters, so check important dates and numbers in the source.")

        corpus = safe("GET", "/corpus", timeout=120)
        if corpus:
            st.subheader("What the files cover")
            stats = corpus["collection"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sentences", f"{stats['sentences']:,}")
            c2.metric("Words", f"{stats['tokens']:,}")
            c3.metric("Distinct words", f"{stats['types']:,}")
            c4.metric("Reduction from stemming", f"{stats['vocabulary_reduction_by_stemming']:.0%}")

            if corpus["topics"]:
                st.write("Topics (k-means on LSA vectors of the passages). These are also what the assistant suggests when it can't answer.")
                st.dataframe(
                    [{"Topic": t["label"], "Passages": t["passages"], "Mostly from": ", ".join(t["documents"])}
                     for t in corpus["topics"]],
                    use_container_width=True, hide_index=True,
                )

            for doc in corpus["documents"]:
                with st.expander(doc["document_name"]):
                    st.write("Keywords: " + ", ".join(doc["keywords"]))
                    for sentence in doc["summary"]:
                        st.write("- " + sentence)
    else:
        st.info("Nothing is indexed yet.")
