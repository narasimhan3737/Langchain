import os
from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from openai import chat
from torch import embedding
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

load_dotenv()

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore


def run_llm(query: str):
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    docsearch = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )
    chat = Ollama(model="deepseek-r1:8b")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    qa = create_retrieval_chain(
        retriever=docsearch.as_retriever(), combine_docs_chain=stuff_documents_chain
    )
    result = qa.invoke(input={"input": query})
    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"],
    }
    return new_result


if __name__ == "__main__":
    res = run_llm(query="What is a LangChain Chain?")
    print(res["answer"])
