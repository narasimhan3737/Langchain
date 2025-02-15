import os
from dotenv import load_dotenv
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import embeddings
from torch import chunk

load_dotenv()

if __name__ == '__main__':
    pdf_path = '/Users/datamotion/Documents/Tutorials/Langchain/Langchain/rag/intro-to-vector-dbs/2210.03629v3.pdf'
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)

    embeddings = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large')
    vectorstore = FAISS.from_documents(docs,embeddings)
    vectorstore.save_local("faiss_index_react")

    
