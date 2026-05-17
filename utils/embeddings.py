
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def load_embedding_model():

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    return embedding_model
