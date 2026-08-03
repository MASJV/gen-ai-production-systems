"""
app.py

This is the Streamlit dashboard. It connects all the pipeline
stages together in the order shown in the architecture diagram:

Upload Files -> Document Readers -> Parsing -> Chunking ->
Embeddings -> Chroma Vector Store -> Hybrid Retrieval ->
BGE Reranker -> Top Chunks -> GPT-4.1 mini -> Final Answer

Run this file with:  streamlit run app.py
"""

import streamlit as st
import config
import document_loader
import chunking
import embeddings_store
import hybrid_retriever
import reranker
import generator

ROBOT_ICON = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' width='1em' height='1em' "
    "fill='none' stroke='currentColor' stroke-width='1.8' stroke-linecap='round' "
    "stroke-linejoin='round' style='vertical-align:-0.15em; display:inline;'>"
    "<rect x='4' y='8' width='16' height='11' rx='2.5'></rect>"
    "<line x1='12' y1='8' x2='12' y2='3'></line>"
    "<circle cx='12' cy='2' r='1'></circle>"
    "<circle cx='9' cy='13.5' r='1'></circle>"
    "<circle cx='15' cy='13.5' r='1'></circle>"
    "<line x1='9' y1='17' x2='15' y2='17'></line>"
    "<line x1='1' y1='12' x2='4' y2='12'></line>"
    "<line x1='20' y1='12' x2='23' y2='12'></line>"
    "</svg>"
)

st.set_page_config(page_title="Enterprise Knowledge Assistant",
                   page_icon=ROBOT_ICON,
                   layout="wide",
                   initial_sidebar_state="expanded")

NAVY = "#0A0E1A"
CYAN = "#00D4FF"
VIOLET = "#7C3AED"
LIGHT_TEXT = "#E5E7EB"

CUSTOM_CSS = """
<style>
.stApp {
    background-color: """ + NAVY + """;
}
h1, h2, h3, h4 {
    color: """ + CYAN + """ !important;
}
.stButton>button {
    background-color: """ + VIOLET + """;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.6em 1.4em;
    font-weight: 600;
}
.stButton>button:hover {
    background-color: """ + CYAN + """;
    color: """ + NAVY + """;
}
[data-testid="stSidebar"] {
    background-color: #10162A;
    text-align: center;
}
[data-testid="stSidebar"] [data-baseweb="tab-list"] {
    justify-content: center;
}
.stMarkdown p, .stMarkdown li, .stMarkdown span, .stMarkdown strong, .stMarkdown em, .stMarkdown code, label {
    color: """ + LIGHT_TEXT + """ !important;
    opacity: 1 !important;
    font-weight: 500;
}

[data-testid="stFileUploaderDropzone"] {
    border: 1px dashed """ + VIOLET + """;
    border-radius: 8px;
    justify-content: center;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: """ + NAVY + """;
}

/* Tabs: recolor the default red/pink active-tab indicator to cyan */
[data-testid="stSidebar"] [data-baseweb="tab-highlight"] {
    background-color: """ + CYAN + """;
}
[data-testid="stSidebar"] button[data-baseweb="tab"][aria-selected="true"] {
    color: """ + CYAN + """;
}
[data-testid="stSidebar"] button[data-baseweb="tab"] p {
    color: #DEB887 !important;
}

/* Card look for the content under each sidebar tab, so it isn't flat text on flat background */
[data-testid="stSidebar"] [data-baseweb="tab-panel"] {
    background-color: #141B33;
    border: 1px solid #232B4D;
    border-radius: 10px;
    padding: 16px 12px;
    margin-top: 10px;
}

/* Source chunk expanders: match the dark card look instead of default light boxes */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background-color: #0F1425;
    border: 1px solid #232B4D;
    border-radius: 8px;
}

/* Slightly tighter top spacing so the sidebar doesn't start with a big empty gap */
[data-testid="stSidebarUserContent"] {
    padding-top: 0rem;
}

[data-testid="stWidgetLabel"] {
    justify-content: center;
}

/* Headers (title, section headers) sit in a flex wrapper that holds the hover
   anchor-link icon, so text-align alone doesn't center them - this does. */
[data-testid="stHeadingWithActionElements"] {
    justify-content: center;
}

/* Headings sometimes wrap their visible text in an inner span, which the
   .stMarkdown span rule above then colors LIGHT_TEXT instead of cyan.
   Targeting every element inside the heading (not just the h-tag) wins that fight. */
[data-testid="stHeadingWithActionElements"] h1,
[data-testid="stHeadingWithActionElements"] h2,
[data-testid="stHeadingWithActionElements"] h3,
[data-testid="stHeadingWithActionElements"] h4,
[data-testid="stHeadingWithActionElements"] h1 *,
[data-testid="stHeadingWithActionElements"] h2 *,
[data-testid="stHeadingWithActionElements"] h3 *,
[data-testid="stHeadingWithActionElements"] h4 * {
    color: """ + CYAN + """ !important;
}

/* Center each source's title inside its expander header in the sidebar */
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    justify-content: center;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary p,
[data-testid="stSidebar"] [data-testid="stExpander"] summary span {
    color: #FFFFFF !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align:center '>" + ROBOT_ICON + " Enterprise Knowledge Assistant</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; font-size:1.05rem;'>"
    "Advanced RAG pipeline: multi-document upload, hybrid search, and reranking."
    "</p>",
    unsafe_allow_html=True
)

# session_state lets us remember things between button clicks
if "collection" not in st.session_state:
    st.session_state.collection = None
if "bm25_index" not in st.session_state:
    st.session_state.bm25_index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "last_sources" not in st.session_state:
    st.session_state.last_sources = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""

# ---------------------------------------------------------
# Sidebar: Upload documents (tab 1) and view sources (tab 2)
# ---------------------------------------------------------
with st.sidebar:
    entered_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="Enter your OpenAI API key"
    )

    if st.button("Set", use_container_width=True):
        if entered_api_key:
            st.session_state.openai_api_key = entered_api_key
            st.success("API key set for this session.")
        else:
            st.warning("Please enter an API key.")

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    st.markdown(
        "<div style='display:flex; justify-content:center; align-items:center; gap:6px; "
        "color:" + LIGHT_TEXT + "; font-weight:700; font-size:1.1rem;'>"
        + ROBOT_ICON + " Document Assistant </div>"
        "<hr style='margin:8px 0 14px 0; border-color:" + CYAN + "; opacity:0.4;'>",
        unsafe_allow_html=True
    )

    upload_tab, sources_tab = st.tabs(["📁 Upload", "📚 Sources"])

    with upload_tab:
        st.markdown("<h2 style='text-align:center'>📁 Upload Documents Here</h2>", unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, TXT, MD, or CSV files",
            accept_multiple_files=True
        )

        if st.button("🚀Process Documents", use_container_width=True):
            if not st.session_state.openai_api_key:
                st.warning("Please set your OpenAI API key first.")
            elif uploaded_files:
                with st.spinner("Reading and processing documents..."):

                    folder_path = document_loader.save_uploaded_files(
                        uploaded_files, config.UPLOAD_FOLDER
                    )
                    documents = document_loader.load_documents(folder_path)

                    chunks = chunking.chunk_documents(
                        documents,
                        config.CHUNK_SIZE,
                        config.CHUNK_OVERLAP
                    )

                    collection = embeddings_store.build_vector_store(chunks, st.session_state.openai_api_key)
                    bm25_index = hybrid_retriever.build_bm25_index(chunks)

                    st.session_state.collection = collection
                    st.session_state.bm25_index = bm25_index
                    st.session_state.chunks = chunks

                st.success("Documents processed Ready to answer questions. Total chunks created: " + str(len(chunks)))
            else:
                st.warning("Please upload at least one file first.")

    with sources_tab:
        st.markdown("<h2 style='text-align:center'>📚 Sources</h2>", unsafe_allow_html=True)
        if st.session_state.last_sources:
            for chunk_info in st.session_state.last_sources:
                box_title = chunk_info["source"] + " - " + chunk_info["id"]
                with st.expander(box_title):
                    st.write(chunk_info["text"])
        else:
            st.info("Ask a question to see the source chunks here.")

    st.markdown("<hr style='margin:20px 0 14px 0; border-color:" + CYAN + "; opacity:0.4;'>",
            unsafe_allow_html=True)

# ---------------------------------------------------------
# Ask a question
# ---------------------------------------------------------
st.markdown(
    "<p style='text-align:center; font-size:1.15rem; font-weight:700;'>"
    "Ask your question below"
    "</p>",
    unsafe_allow_html=True
)

question = st.text_area(
    "Question",
    placeholder="Type your question about the uploaded documents: ",
    height=100
)

if st.button("🔍 Get Answer", use_container_width=True):
    if not st.session_state.openai_api_key:
        st.warning("Please set your OpenAI API key first.")
    elif st.session_state.collection is None:
        st.warning("Please process documents first.")
    elif question.strip() == "":
        st.warning("Please type a question.")
    else:
        with st.spinner("Searching documents and generating answer..."):

            fused_chunks = hybrid_retriever.hybrid_search(
                st.session_state.collection,
                st.session_state.bm25_index,
                st.session_state.chunks,
                question,
                st.session_state.openai_api_key
            )

            top_chunks = reranker.rerank_chunks(
                question,
                fused_chunks,
                config.TOP_K_FINAL
            )

            answer = generator.generate_answer(question, top_chunks, st.session_state.openai_api_key)
            st.session_state.last_sources = top_chunks
            st.session_state.last_answer = answer

        st.rerun()

if st.session_state.last_answer:
    st.subheader("Answer")
    st.write(st.session_state.last_answer)