import os
from dotenv import load_dotenv
from langchain import hub
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
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

    new_vectorstore = FAISS.load_local(
        "faiss_index_react", embeddings, allow_dangerous_deserialization=True
    )

    llm = Ollama(model="deepseek-r1:8b")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    retrieval_chain = create_retrieval_chain(
        new_vectorstore.as_retriever(), combine_docs_chain
    )

    res = retrieval_chain.invoke({"input": "Give me the gist of ReAct in 3 sentences"})
    print(res["answer"])