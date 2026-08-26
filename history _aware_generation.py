import os
from dotenv import load_dotenv
from groq import Groq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
# AIMessage add kiya hai history store karne ke liye
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

# 1. Groq ke active models fetch aur strictly clean chat models filter karna
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
all_models = [m.id for m in client.models.list().data]

# Third-party, gated aur non-chat models ko filter out karna
clean_chat_models = [
    m for m in all_models 
    if not any(bad in m.lower() for bad in ["guard", "whisper", "audio", "vision", "canopy", "orpheus", "/"])
]

print("Tumhare account par available pure Chat models:")
for m in clean_chat_models:
    print(f"- {m}")

# Substring matching: inme se jo bhi match ho jaye pick kar lo
keywords = ["llama-3.3", "llama-3.1", "gemma", "mixtral", "qwen", "llama"]

selected_model = None
for kw in keywords:
    for m in clean_chat_models:
        if kw in m.lower():
            selected_model = m
            break
    if selected_model:
        break

# Safe Fallback
if not selected_model and clean_chat_models:
    selected_model = clean_chat_models[0]

print(f"\n--> Auto-Selected Chat Model: '{selected_model}'\n")

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

# 3. Model Initialization
model = ChatGroq(
    model=selected_model,
    temperature=0.1
)

# Chat history store karne ke liye array
chat_history = []

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")
    
    # Step 1: History-Aware Question Generation
    if chat_history:
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]
        
        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching DB for rewritten query: '{search_question}'")
    else:
        search_question = user_question
        print(f"Searching DB for query: '{search_question}'")

    # Step 2: Search for relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(search_question)
    
    if not relevant_docs:
        print("No relevant documents found. Skipping LLM call.")
        return
    else:
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")
            print("-" * 50)

    # Step 3: Context Preparation
    context_text = "\n\n".join([f"- {doc.page_content}" for doc in relevant_docs])

    system_prompt = (
        "You are a helpful assistant. "
        "Please answer the question using ONLY the provided context documents. "
        "If the answer is not present in the documents, respond with 'I don't know'."
    )
    
    human_prompt = f"Documents:\n{context_text}\n\nQuestion: {user_question}"

    # Step 4: Final LLM Call (Includes System Prompt, History, and Current Context)
    messages = [
        SystemMessage(content=system_prompt),
    ] + chat_history + [
        HumanMessage(content=human_prompt)
    ]

    print(f"\nSending query and context to Groq LLM ({selected_model})...")

    try:
        result = model.invoke(messages)
        answer = result.content
        
        # Step 5: Update History Memory
        chat_history.append(HumanMessage(content=user_question))
        chat_history.append(AIMessage(content=answer))
        
        print("\n--- Generated AI Response ---")
        print(answer)
        print("-----------------------------")
        return answer
    except Exception as e:
        print(f"\n[Error] LLM Call Failed: {e}")
        print("Hint: Make sure GROQ_API_KEY is properly set in your .env file.")
        return None

def start_chat():
    print("\n==============================================")
    print("Ask me questions! Type 'quit' or 'exit' to end.")
    print("==============================================\n")
    
    while True:
        question = input("Your question: ")
        
        if question.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
            
        if question.strip():
            ask_question(question)

if __name__ == "__main__":
    start_chat()