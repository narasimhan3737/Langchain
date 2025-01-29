import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone, PineconeEmbeddings, PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import pinecone


load_dotenv()
if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader("E:/Work/LLM/LangChain/rag/intro-to-vector-dbs/mediumblog1.txt",encoding='unicode_escape')    
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")
    
    pc = pinecone.Pinecone(api_key=os.environ['PINECONE_API_KEY'])
    #pinecone_client = Pinecone(api_key=os.environ['PINECONE_API_KEY'], environment=os.environ['PINECONE_ENV'])

    
    index_name = os.environ.get("INDEX_NAME2")
    
    """
    if index_name not in pc.list_indexes():
        pc.create_index(
            name=index_name,
            dimension=384,  # Match this to the embedding model's output dimension
            metric="cosine",  # You can also use "euclidean" or others
            spec=ServerlessSpec(
                cloud="aws",  # Specify your cloud provider
                region=os.environ['PINECONE_ENV']  # Your Pinecone region
            )                        
        )
        #raise ValueError(f"Index '{index_name}' does not exist. Please create it in your Pinecone console.")
    """
    
    #index = pinecone.get_index(index_name)
    index = pc.Index(host=os.environ.get("INDEX_HOST2"))
    index_stats = index.describe_index_stats()
    index_dim = index_stats['dimension']

    

    #embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPEN_API_KEY"),model="text-embedding-3-small")
    #embeddings = OllamaEmbeddings(model="llama3")
    #embeddings = PineconeEmbeddings(model="multilingual-e5-large")
    #embeddings = HuggingFaceEmbeddings(model_name='Xenova/gte-small')
    
    if index_dim == 384:
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    elif index_dim == 1024:
        embeddings = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large')
        
    else:
        raise ValueError(f"Unsupported index dimension: {index_dim}")
    
    print("ingesting....")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ['INDEX_NAME2'])