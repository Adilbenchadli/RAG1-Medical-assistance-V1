
import streamlit as st
import os
from llama_cpp import Llama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from huggingface_hub import hf_hub_download

# --- Configuration --- #
LLM_MODEL_NAME_OR_PATH = "TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF"
LLM_MODEL_BASENAME = "capybarahermes-2.5-mistral-7b.Q5_0.gguf" # Use the Q5_0 version as it's the one that was loaded initially
EMBEDDING_MODEL_NAME = "stsb-mpnet-base-v2"
VECTOR_DB_DIR = "medical_db"

# --- Load LLM Model --- #
@st.cache_resource
def load_llm():
    model_path = hf_hub_download(
        repo_id=LLM_MODEL_NAME_OR_PATH,
        filename=LLM_MODEL_BASENAME
    )
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=36,
        n_batch=512
    )
    return llm

# --- Load Embedding Model --- #
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# --- Load Vector Store --- #
@st.cache_resource
def load_vectorstore(_embedding_model): # Changed embedding_model to _embedding_model
    return Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=_embedding_model) # Changed embedding_model to _embedding_model

# --- RAG Components --- #
llm = load_llm()
embedding_model = load_embedding_model()
vectorstore = load_vectorstore(embedding_model)

qna_system_message = (
    "You are an expert medical assistant. Answer the user's question accurately, thoroughly, "
    "and strictly using ONLY the provided context from the medical manual. "
    "Instructions:\n"
    "1. Do not extrapolate or use outside knowledge.\n"
    "2. If the answer cannot be found in the context, state exactly: 'Information not available in the provided manual.'\n"
    "3. Explicitly retain the source and page citations present at the start of the context chunks."
)

qna_user_message_template = """[START OF TRUSTED CONTEXT]
{context}
[END OF TRUSTED CONTEXT]

Clinical Question: {question}

Structured Answer:"""

def generate_rag_response(user_input, max_tokens=200, temperature=0.2, search_type="similarity", k=5, lambda_mult=0.7):
    search_kwargs = {'k': k}
    if search_type == 'mmr':
        search_kwargs['fetch_k'] = 10
        search_kwargs['lambda_mult'] = lambda_mult

    current_retriever = vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

    relevant_document_chunks = current_retriever.invoke(user_input)
    context_list = [d.page_content for d in relevant_document_chunks]
    context_for_query = "\n\n---\n\n".join(context_list)

    user_message = qna_user_message_template.replace('{context}', context_for_query)
    user_message = user_message.replace('{question}', user_input)

    full_prompt = f"[INST] {qna_system_message}\n\n{user_message} [/INST]"

    try:
        model_output = llm(
            prompt=full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            top_k=40
        )
        response_text = model_output['choices'][0]['text'].strip()
    except Exception as e:
        response_text = f"Sorry, I encountered the following error: \n {e}"

    return response_text

# --- Streamlit UI --- #
st.title("🩺 Medical RAG Assistant")
st.write("Ask any medical question, and I'll provide an answer based on the Merck Manuals.")

user_question = st.text_area("Enter your medical question here:", height=100)

if st.button("Get Answer"): # Here I use the original generate_rag_response_retrever_choice function with the optimal configuration values
    if user_question:
        with st.spinner("Searching and generating response..."):
            response = generate_rag_response(user_question, max_tokens=200, temperature=0.0, search_type="similarity", k=5)
            st.markdown(f"**Answer:** {response}")
    else:
        st.warning("Please enter a question.")
