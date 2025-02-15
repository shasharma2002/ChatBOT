from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
import os

class VectorStoreManager:
    def __init__(self, embeddings: OpenAIEmbeddings, chroma_dir: str):
        self.embeddings = embeddings
        self.chroma_dir = chroma_dir

    def setup_vector_store(self, test_case_dir: str) -> Chroma:
        vectorstore = None
        if not os.path.exists(test_case_dir) or not os.listdir(test_case_dir):
            vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.chroma_dir
            )
        return vectorstore
