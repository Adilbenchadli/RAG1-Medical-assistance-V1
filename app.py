
import streamlit as st
import os
from llama_cpp import Llama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download
import chromadb


# 1. Config & Paths
HF_REPO_ID = "smflaser/medical-chroma-db" # Replace with your dataset repo
LOCAL_DB_PATH = "medical_db"

LLM_MODEL_NAME_PATH = "TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF"
LLM_MODEL_BASENAME = "capybarahermes-2.5-mistral-7b.Q5_0.gguf"
EMBEDDING_MODEL_NAME = "stsb-mpnet-base-v2"

# ==========================================
# 2. Resource Loaders (Cached)
# ==========================================
@st.cache_resource
def load_vectorstore_from_hf():
    """Downloads the Chroma vector database from HF if not existing, then loads Chroma."""
    if not os.path.exists(LOCAL_DB_PATH) or not os.listdir(LOCAL_DB_PATH):
        st.info("Downloading vector database from Hugging Face Hub...")
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=LOCAL_DB_PATH,
            token=st.secrets.get("HuggingFace", None)
        )
        st.success("Database download complete!")
    
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma(
        persist_directory=LOCAL_DB_PATH,
        embedding_function=embedding_model
    )
    return vectorstore

@st.cache_resource
def load_llm():
    """Downloads and loads the quantized Llama GGUF model."""
    model_path = hf_hub_download(
        repo_id=LLM_MODEL_NAME_PATH,
        filename=LLM_MODEL_BASENAME
    )
    # n_threads set to CPU physical core count (e.g., 4 or 8) for optimum inference speed
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,           # Reduced from 32768/4096 to lower memory and speed up CPU processing
        n_threads=os.cpu_count() or 4, 
        n_gpu_layers=0,       # Set to >0 if running on CUDA/GPU environment
        n_batch=512
    )
    return llm

# ==========================================
# 3. Prompts & RAG Logic
# ==========================================
qna_system_message = (
    "You are an expert medical assistant. Answer the user's question accurately, thoroughly, "
    "and strictly using ONLY the provided content from the medical manual.\n"
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

def generate_rag_response(user_input, vectorstore, llm, max_tokens=300, temperature=0.1, k=5):
    """Retrieves relevant context using vector similarity search and generates an answer."""
    # Use standard similarity search with top-k=5 to capture full medical context
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
    relevant_docs = retriever.invoke(user_input)
    context_list = [d.page_content for d in relevant_docs]
    context_for_query = "\n\n".join(context_list)
    
    user_message = qna_user_message_template.replace("{context}", context_for_query)
    user_message = user_message.replace("{question}", user_input)
    
    # Formatted prompt following Mistral Chat Template
    full_prompt = f"<|im_start|>system\n{qna_system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
    
    try:
        model_output = llm(
            prompt=full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            stop=["<|im_end|>", "Clinical Question:"]
        )
        response_text = model_output['choices'][0]['text'].strip()
    except Exception as e:
        response_text = f"An error occurred during response generation: {str(e)}"
        
    return response_text

# ==========================================
# 4. Streamlit Application User Interface
# ==========================================
st.set_page_config(page_title="Medical RAG Assistant", layout="wide")
st.title("🩺 Medical RAG Assistant")
st.write("Ask any medical question. Answers are derived strictly from the Merck Manual.")

# Load cached resources
try:
    vectorstore = load_vectorstore_from_hf()
    llm = load_llm()
    st.sidebar.success("Models and VectorDB successfully loaded!")
except Exception as e:
    st.error(f"Initialization error: {e}")
    st.stop()

# Input area
user_question = st.text_area("Enter your medical question:", height=100, 
                             placeholder="e.g., What is the protocol for managing sepsis in a critical care unit?")

if st.button("Get Answer"):
    if user_question.strip():
        with st.spinner("Searching Merck Manual & generating response..."):
            response = generate_rag_response(user_question, vectorstore, llm)
            st.markdown("### Answer")
            st.write(response)
    else:
        st.warning("Please enter a clinical question.")
