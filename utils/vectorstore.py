import os

from langchain.vectorstores import FAISS


VECTOR_DB_PATH = "data/faiss_index"


def create_vector_store(documents, embedding_model):

    vector_store = FAISS.from_documents(
        documents,
        embedding_model
    )

    vector_store.save_local(VECTOR_DB_PATH)

    return vector_store


def load_vector_store(embedding_model):

    if os.path.exists(VECTOR_DB_PATH):

        vector_store = FAISS.load_local(
            VECTOR_DB_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )

        return vector_store

    return None
