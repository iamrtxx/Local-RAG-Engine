import os
from dotenv import load_dotenv
from groq import Groq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# 1. Fetch all active models dynamically from your Groq account
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
all_models = [m.id for m in client.models.list().data]

print("Tumhare account par active Groq models:")
for m_id in all_models:
    print(f"- {m_id}")

# Auto-select: Jo bhi working model list mein mile usay automatically pick kar lo
preferred_models = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "mixtral-8x7b-32768"
]

selected_model = None
for pref in preferred_models:
    if pref in all_models:
        selected_model = pref
        break

# Fallback agar preferred list match na ho toh pehla model utha lo
if not selected_model and all_models:
    selected_model = all_models[0]

print(f"\n--> Selected Active Model for Generation: '{selected_model}'\n")

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
query = "What was Microsoft's first hardware product release?"

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

    # 4. Context Preparation
    context_text = "\n\n".join([f"- {doc.page_content}" for doc in relevant_docs])

    system_prompt = (
        "You are a helpful assistant. "
        "Please answer the question using ONLY the provided context documents. "
        "If the answer is not present in the documents, respond with 'I don't know'."
    )
    
    human_prompt = f"Documents:\n{context_text}\n\nQuestion: {query}"

    # 5. Dynamic LLM Call
    model = ChatGroq(
        model=selected_model,
        temperature=0.1
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]

    print(f"\nSending query and context to Groq LLM ({selected_model})...")

    try:
        result = model.invoke(messages)
        print("\n--- Generated AI Response ---")
        print(result.content)
        print("-----------------------------")
    except Exception as e:
        print(f"\n[Error] LLM Call Failed: {e}")
        print("Hint: Make sure GROQ_API_KEY is properly set in your .env file.")