
import streamlit as st
import os
import fitz

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import FAISS

from langchain.memory import ConversationBufferMemory

from langchain.chains import ConversationalRetrievalChain

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain.docstore.document import Document


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Context-Aware RAG Chatbot")
st.markdown("Upload PDFs and chat with your documents using Gemini AI")


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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None


# =========================
# PDF UPLOAD
# =========================

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF Files",
    type=["pdf"],
    accept_multiple_files=True
)


# =========================
# PROCESS DOCUMENTS
# =========================

if st.sidebar.button("Process PDFs"):

    if uploaded_files and GOOGLE_API_KEY:

        with st.spinner("Processing PDFs..."):

            all_text = ""

            for uploaded_file in uploaded_files:

                pdf_document = fitz.open(
                    stream=uploaded_file.read(),
                    filetype="pdf"
                )

                for page_num in range(len(pdf_document)):

                    page = pdf_document[page_num]

                    text = page.get_text()

                    all_text += text + "\n"


            # =========================
            # CHUNKING
            # =========================

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = text_splitter.split_text(all_text)

            documents = [
                Document(page_content=chunk)
                for chunk in chunks
            ]


            # =========================
            # EMBEDDINGS
            # =========================

            embedding_model = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001"
            )


            # =========================
            # VECTOR STORE
            # =========================

            vector_store = FAISS.from_documents(
                documents,
                embedding_model
            )


            # =========================
            # RETRIEVER
            # =========================

            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )


            # =========================
            # MEMORY
            # =========================

            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )


            # =========================
            # GEMINI MODEL
            # =========================

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.3
            )


            # =========================
            # RAG CHAIN
            # =========================

            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                return_source_documents=True
            )

            st.session_state.qa_chain = qa_chain

            st.success("PDFs processed successfully!")

    else:
        st.warning("Please upload PDFs and enter API Key")


# =========================
# CHAT INTERFACE
# =========================

# DISPLAY CHAT HISTORY

for role, message in st.session_state.chat_history:
    with st.chat_message(role):
         st.markdown(message)

query = st.chat_input("Ask a question from your PDFs...")


if query and st.session_state.qa_chain:

    # DISPLAY USER MESSAGE

    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Generating Response..."):

        response = st.session_state.qa_chain.invoke(
            {"question": query}
        )

        answer = response["answer"]

        source_docs = response["source_documents"]


    # DISPLAY ASSISTANT RESPONSE

    with st.chat_message("assistant"):

        # DISPLAY ANSWER
        st.markdown(answer)

        # COLLAPSIBLE SOURCE CITATIONS
        with st.expander("📚 Show Source Citations"):

            for i, doc in enumerate(source_docs):

                st.markdown(f"### Source {i+1}")

                st.write(
                    doc.page_content[:500] + "..."
                )


    # SAVE CHAT HISTORY

    st.session_state.chat_history.append(
        ("user", query)
    )

    st.session_state.chat_history.append(
        ("assistant", answer)
    )
