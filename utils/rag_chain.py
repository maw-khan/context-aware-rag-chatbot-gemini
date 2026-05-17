
from langchain.memory import ConversationBufferMemory

from langchain.chains import ConversationalRetrievalChain

from langchain_google_genai import ChatGoogleGenerativeAI


def create_rag_chain(retriever):

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

    return qa_chain
