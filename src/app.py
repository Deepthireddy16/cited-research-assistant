"""
app.py
Streamlit UI for the cited research assistant. Wraps retrieval.py +
generate.py in a simple web interface: type a question, get a
grounded answer with inline [Source N] citations and a verified
references list underneath.

Run with:
    streamlit run src/app.py
"""

import re
import streamlit as st
from retrieval import Retriever
from generate import answer_question, build_reference_line

st.set_page_config(page_title="Cited Research Assistant", page_icon="📚")


@st.cache_resource
def load_retriever():
    """
    Cached so the embedding model, Chroma connection, and BM25 index
    are built ONCE per app session, not re-built on every question —
    Streamlit reruns the whole script on every interaction, so without
    this, each click would reload everything from scratch.
    """
    return Retriever()


st.title("📚 Cited Research Assistant")
st.markdown(
    "Ask a question about **transformer attention mechanisms**. "
    "Answers are generated only from a corpus of 19 arXiv papers, "
    "with citations linking back to the real paper, author, and page."
)

retriever = load_retriever()

question = st.text_input("Your question:", placeholder="e.g. How does self-attention work?")
ask_clicked = st.button("Ask", type="primary")

if ask_clicked and question.strip():
    with st.spinner("Retrieving sources and generating an answer..."):
        answer, references, chunks = answer_question(question, retriever)

    st.markdown("### Answer")
    st.markdown(answer)

    # Only show references the model actually cited (e.g. "[Source 2]")
    # — not every retrieved chunk. Showing all 5 regardless of whether
    # they were used makes a refusal ("I don't have that information")
    # look like it was informed by 5 papers, which is misleading.
    cited_numbers = sorted(set(int(n) for n in re.findall(r"\[Source\s*(\d+)", answer)))

    if cited_numbers:
        st.markdown("### References")
        for i in cited_numbers:
            st.markdown(f"- {build_reference_line(chunks[i - 1], i)}")
    else:
        st.caption("No sources were cited for this answer.")

elif ask_clicked:
    st.warning("Type a question first.")

st.markdown("---")
st.caption(
    "Runs entirely locally: BAAI/bge-small-en-v1.5 embeddings, "
    "hybrid vector + BM25 retrieval, and llama3.1:8b via Ollama. "
    "No paid APIs."
)
