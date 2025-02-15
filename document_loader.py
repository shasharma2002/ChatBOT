import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

class DocumentLoader:
    def __init__(self, test_case_dir: str, chroma_dir: str):
        self.test_case_dir = test_case_dir
        self.chroma_dir = chroma_dir
        self.embeddings = OpenAIEmbeddings()

    def load_new_documents(self, processed_files: set) -> bool:
        new_documents = []

        for file in os.listdir(self.test_case_dir):
            file_path = os.path.join(self.test_case_dir, file)
            if file_path in processed_files:
                continue

            try:
                if file.endswith('.txt'):
                    loader = TextLoader(file_path)
                    new_documents.extend(loader.load())
                elif file.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    new_documents.extend(loader.load())
                processed_files.add(file_path)
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")

        if new_documents:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(new_documents)

            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=self.chroma_dir
            )
            return vectorstore, True
        return None, False
