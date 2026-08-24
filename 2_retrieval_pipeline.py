import os
from dotenv import load_dotenv
from langchain_chroma import Chroma

# OpenAI ki jagah Local HuggingFace Embeddings import kiye hain
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

persistent_directory = "db/chroma_db"

print("Loading local embedding model...")
# Same model use kiya hai jo ingestion_pipeline.py mein use kiya tha
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Connecting to ChromaDB...")
# Vector database load kar rahe hain
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

# Search for relevant documents
query = "What was NVIDIA's first graphics accelerator called?"

print(f"\nRunning semantic search for: '{query}'")
retriever = db.as_retriever(search_kwargs={"k": 5})

# Note: Agar exact matching chahiye toh similarity threshold wala code uncomment kar sakte ho
# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3  # Only return chunks with cosine similarity ≥ 0.3
#     }
# )

relevant_docs = retriever.invoke(query)

print(f"\nUser Query: {query}")
print("--- Context Found ---")

# Agar documents nahi milte toh graceful handle kiya hai
if not relevant_docs:
    print("No relevant documents found. (Cosine similarity might be too low or data doesn't exist).")
else:
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
        print("-" * 50)


# Synthetic Questions: 
# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"