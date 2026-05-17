from langchain_community.document_loaders import PyPDFLoader
import tempfile


def load_pdf_documents(uploaded_files):

    documents = []

    for uploaded_file in uploaded_files:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:

            tmp_file.write(uploaded_file.read())

            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)

        docs = loader.load()

        documents.extend(docs)

    return documents
