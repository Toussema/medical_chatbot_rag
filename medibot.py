import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA  # Fixed import
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from dotenv import load_dotenv, find_dotenv  # Uncommented for env loading
load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def main():
    st.title("Ask Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """
        
        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGroq(
                    model="llama-3.3-70b-versatile",  # Updated model name
                    temperature=0.0,
                    groq_api_key=os.environ["GROQ_API_KEY"],
                ),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})

            result = response["result"].strip()
            source_documents = response["source_documents"]

            # Check if it's a "don't know" response to hide sources
            if "don't know" in result.lower() or "no information" in result.lower():
                result_to_show = result  # No sources
            else:
                # Format the main answer (e.g., detect lists and bullet them)
                if "symptoms" in prompt.lower() and "include" in result.lower():  # Example for symptoms—expand as needed
                    symptoms = result.split(", ")
                    formatted_result = "**Symptoms:**\n" + "\n".join([f"- {symptom.strip()}" for symptom in symptoms])
                else:
                    formatted_result = result  # Default to plain text

                result_to_show = formatted_result

                # Add sources in a collapsible expander (more ergonomic)
                with st.chat_message('assistant'):
                    st.markdown(result_to_show)
                    with st.expander("View Sources"):
                        for i, doc in enumerate(source_documents):
                            st.write(f"**Source {i+1}:**")
                            st.write(f"- Page: {doc.metadata.get('page_label', 'N/A')}")
                            st.write(f"- File: {doc.metadata.get('source', 'N/A')}")
                            st.write(f"- Content Snippet: {doc.page_content[:200]}...")  # Truncated for brevity

                # Append the full result to session state (including sources for history)
                st.session_state.messages.append({'role': 'assistant', 'content': result_to_show + "\n\n(Sources hidden in expander)"})
                return  # Exit early since we handled display

            # Display simple response
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()