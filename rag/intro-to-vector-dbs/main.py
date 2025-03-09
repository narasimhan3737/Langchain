import os


from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import Pinecone, PineconeEmbeddings, PineconeVectorStore
from dotenv import load_dotenv
from langchain import hub

load_dotenv()

if __name__ == "__main__":
    print("Retrieving...")

    # embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceEmbeddings()
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    # llm = ChatOpenAI()
    llm = Ollama(model="deepseek-r1:8b")

    query = "what is Pinecone in machine learning?"
    chain = PromptTemplate.from_template(template=query) | llm
    result = chain.invoke(input={})
    # print(result.content)

    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME2"], embedding=embeddings
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    result = retrieval_chain.invoke(input={"input": query})
    print(result)
