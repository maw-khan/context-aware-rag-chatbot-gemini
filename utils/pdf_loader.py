%%writefile pdf_loader.py

import fitz
from langchain.docstore.document import Document


def load_pdf_documents(uploaded_files):

    documents = []

    for uploaded_file in uploaded_files:

        pdf_document = fitz.open(
            stream=uploaded_file.read(),
            filetype="pdf"
        )

        for page_num in range(len(pdf_document)):

            page = pdf_document[page_num]

            text = page.get_text()

            if text.strip():

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": uploaded_file.name,
                            "page": page_num + 1
                        }
                    )
                )

    return documents
