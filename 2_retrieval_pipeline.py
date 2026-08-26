import os
from dotenv import load_dotenv
from groq import Groq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()



# 2. Vector DB & Local Embeddings Setup
persistent_directory = "db/chroma_db"

print("Loading local embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Connecting to ChromaDB...")
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

# 3. Search Query
query = "tell me about NVIDIA's latest GPU architecture."

print(f"\nRunning semantic search for: '{query}'")
retriever = db.as_retriever(search_kwargs={"k": 5})

relevant_docs = retriever.invoke(query)

print(f"\nUser Query: {query}")
print("--- Context Found ---")

if not relevant_docs:
    print("No relevant documents found. Skipping LLM call.")
else:
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
        print("-" * 50)

  

