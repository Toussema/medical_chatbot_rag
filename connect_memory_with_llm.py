import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain  # Updated import
from langchain_classic.chains.combine_documents import create_stuff_documents_chain  # Updated import
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# ----------------------------
# Step 1: Setup Groq LLM
# ----------------------------
load_dotenv()
def load_llm():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        max_tokens=1024
    )
    return llm


# ----------------------------
# Step 2: Custom Prompt
# ----------------------------

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, say you don't know — do not invent anything.
Answer ONLY from the context.

Context:
{context}

Question:
{input}

Start the answer immediately, no small talk.
"""

def set_custom_prompt(custom_prompt_template):
    return ChatPromptTemplate.from_template(custom_prompt_template)


# ----------------------------
# Step 3: Load Vector DB
# ----------------------------

DB_FAISS_PATH = "vectorstore/db_faiss"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)


# ----------------------------
# Step 4: Build RAG Retrieval Chain
# ----------------------------

llm = load_llm()
retriever = db.as_retriever(search_kwargs={'k': 3})

prompt = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# ----------------------------
# Step 5: Interactive Query
# ----------------------------

user_query = input("Write Query Here: ")

response = retrieval_chain.invoke({'input': user_query})

print("\nRESULT:\n", response["answer"])
print("\nSOURCE DOCUMENTS:\n", response["context"])