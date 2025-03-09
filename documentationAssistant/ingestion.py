from langchain.text_splitter import TextSplitter
from langchain.document_loaders import ReadTheDocsLoader
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama import embeddings

load_dotenv()


def ingest_docs():
    loader = ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest")
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)

    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to Pinecone")
    PineconeVectorStore.from_documents(
        documents, embeddings, index_name="langchain-doc-index"
    )
    print("****Loading to vectorstore done ***")


if __name__ == "__main__":
    ingest_docs()
