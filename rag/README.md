# Medium Analyzer using RAG

A powerful tool for analyzing and retrieving information from text using a combination of advanced techniques like `TextLoader`, `TextSplitter`, `OpenAIEmbeddings`, and `Pinecone` for efficient data processing. The project also integrates `FAISS` as a local vector store, enabling interactive PDF querying and information retrieval with `LCEL` chains.

---

## 🚀 Features

- **Text Loading & Splitting:** Process large text files with `TextLoader` and split them into manageable chunks with `TextSplitter`.  
- **Embeddings & Vector Search:** Generate high-dimensional embeddings using `OpenAIEmbeddings` and store/query them using `Pinecone` or `FAISS` for fast similarity search.  
- **PDF Querying:** Load and analyze PDFs with interactive querying capabilities.  
- **RAG (Retrieval-Augmented Generation):** Use LCEL chains to retrieve and synthesize information, enhancing analysis accuracy and depth.  
- **Local & Remote Storage:** Switch between local (`FAISS`) and cloud-based (`Pinecone`) vector databases for flexible storage options.  

---

## 🛠️ Tech Stack

- **LangChain** — For chaining different components together (loading, splitting, retrieval).  
- **OpenAI Embeddings** — To convert text into vector representations.  
- **Pinecone** — For scalable, cloud-based vector search.  
- **FAISS (Facebook AI Similarity Search)** — For fast, local vector indexing and similarity matching.  
- **LCEL Chains** — To create complex information retrieval workflows.  

---

## 📂 Installation

1. Clone the repository:  

```bash
git clone https://github.com/narasimhan3737/Langchain.git
cd rag
```

2. Set up a virtual environment (optional but recommended):  

```bash
python -m venv venv  
source venv/bin/activate   # On Windows use: venv\Scripts\activate
```

3. Install dependencies:  

```bash
pip install -r requirements.txt
```

4. Configure your environment variables (e.g., OpenAI API key, Pinecone API key):  

Create a `.env` file and add the necessary keys:  

```bash
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

---

## ▶️ Usage

Run the analyzer locally:  

```bash
python main.py
```

Example usage — query a PDF:  

```bash
Enter your query: "Explain the key concepts in the document."
```

The tool will retrieve the most relevant text chunks and provide a summarized response using RAG.

