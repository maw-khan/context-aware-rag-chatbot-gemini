
import os
import streamlit as st

from utils.pdf_loader import load_pdf_documents
from utils.chunking import split_documents
from utils.embeddings import load_embedding_model
from utils.vectorstore import (
    create_vector_store,
    load_vector_store
)
from utils.rag_chain import create_rag_chain


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Context-Aware RAG Chatbot")

st.markdown(
    "Upload PDFs and chat with your documents using Gemini AI"
)


# =========================
# LOAD ENV VARIABLES
# =========================

load_dotenv()


# =========================
# API KEY
# =========================

GOOGLE_API_KEY = st.sidebar.text_input(
    "Enter Gemini API Key",
    type="password"
)

if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# =========================
# SESSION STATE
# =========================

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# =========================
# FILE UPLOAD
# =========================

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF Files",
    type=["pdf"],
    accept_multiple_files=True
)


# =========================
# PROCESS PDFs
# =========================

if st.sidebar.button("Process PDFs"):

    if not GOOGLE_API_KEY:
        st.warning("Please enter Gemini API Key")

    elif not uploaded_files:
        st.warning("Please upload PDF files")

    else:

        try:

            with st.spinner("Processing PDFs..."):

                # LOAD DOCUMENTS
                documents = load_pdf_documents(
                    uploaded_files
                )

                # SPLIT DOCUMENTS
                split_docs = split_documents(
                    documents
                )

                # LOAD EMBEDDINGS
                embedding_model = load_embedding_model()

                # CREATE VECTOR STORE
                vector_store = create_vector_store(
                    split_docs,
                    embedding_model
                )

                # CREATE RETRIEVER
                retriever = vector_store.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 4}
                )

                # CREATE RAG CHAIN
                qa_chain = create_rag_chain(
                    retriever
                )

                st.session_state.qa_chain = qa_chain

                st.success(
                    "PDFs processed successfully!"
                )

        except Exception as e:

            st.error(f"Error: {str(e)}")


# =========================
# CHAT HISTORY
# =========================

for role, message in st.session_state.chat_history:

    with st.chat_message(role):

        st.markdown(message)


# =========================
# USER INPUT
# =========================

query = st.chat_input(
    "Ask questions from your PDFs..."
)


# =========================
# GENERATE RESPONSE
# =========================

if query and st.session_state.qa_chain:

    with st.chat_message("user"):

        st.markdown(query)

    try:

        with st.spinner("Generating response..."):

            response = st.session_state.qa_chain.invoke(
                {"question": query}
            )

            answer = response["answer"]

            source_docs = response[
                "source_documents"
            ]

        with st.chat_message("assistant"):

            st.markdown(answer)

            with st.expander(
                "📚 Source Citations"
            ):

                for i, doc in enumerate(source_docs):

                    source = doc.metadata.get(
                        "source",
                        "Unknown"
                    )

                    page = doc.metadata.get(
                        "page",
                        "N/A"
                    )

                    st.markdown(
                        f"""
### Source {i+1}

**Document:** {source}

**Page:** {page}
"""
                    )

                    st.write(
                        doc.page_content[:500] + "..."
                    )

        st.session_state.chat_history.append(
            ("user", query)
        )

        st.session_state.chat_history.append(
            ("assistant", answer)
        )

    except Exception as e:

        st.error(f"Error: {str(e)}")
