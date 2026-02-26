import streamlit as st
import tempfile
import hashlib
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
groq_api_key = os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader , TextLoader, Docx2txtLoader , UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader, SeleniumURLLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from urllib.parse import urlparse

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Document & Web Assistant",
    page_icon="📄",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align: center;'>📄 AI Document & Web Assistant</h1>",
    unsafe_allow_html=True
)


# ---------------------------------------------------
# SIDEBAR SETTINGS
# ---------------------------------------------------
with st.sidebar:

    st.header("⚙️ Settings")

    user_options = st.selectbox(
        "Select what you want to upload",
        options=["Document" , "Webpage"]
    )

    if user_options == "Document":
        uploaded_file = st.file_uploader(
            "Upload PDF",
            type=["pdf" , 'txt' , 'docx', 'pptx'],
            accept_multiple_files=True
        )

        if uploaded_file:
            total_size = sum([file.size for file in uploaded_file])
            if total_size > 30 * 1024 * 1024:  # 30MB limit
                st.sidebar.error(f"Total upload size exceeds 30MB! Current size: {total_size / (1024*1024):.2f} MB")
                uploaded_file = None  # prevent further processing
                st.stop()

            for file in uploaded_file:
                if file.size > 30 * 1024 * 1024:  # 30MB limit per file
                    st.sidebar.error(f"File {file.name} exceeds 30 MB!")
                    uploaded_file = None
                    st.stop()

    elif user_options == "Webpage":
        url = st.text_input(
            "Enter website URL",
            placeholder="https://example.com"
        )

    initialize = st.button("🚀 Initialize Chatbot")
    clear_chat = st.button("🗑 Clear Chat")

# Ensure variables always exist
uploaded_file = locals().get("uploaded_file", None)
url = locals().get("url", None)


# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

# Clear chat
if clear_chat:
    st.session_state.messages = []
    st.session_state.chat_history = ChatMessageHistory()
    st.success("Chat cleared.")

# ---------------------------------------------------
# CACHED EMBEDDINGS
# --------------------------------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEndpointEmbeddings(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HF_TOKEN")
    )

#Loading and handling different file types
def load_document(uploaded_file):
    try:
        file_type = uploaded_file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp:
            tmp.write(uploaded_file.read())
            file_path = tmp.name

        loader_map = {
            "pdf": PyPDFLoader,
            "txt": TextLoader,
            "docx": Docx2txtLoader,
            "pptx": UnstructuredPowerPointLoader
        }
        
        if file_type not in loader_map:
            st.error(f"Unsupported file type: {file_type}")
            return []
        
        try:
            loader = loader_map[file_type](file_path)
            docs = loader.load()
        finally:
            # Cleanup temp file immediately
            os.remove(file_path)
        
        if not docs:
            st.warning(f"File {uploaded_file.name} appears to be empty or unreadable.")
        return docs
    except Exception as e:
        st.error(f"Error loading {uploaded_file.name}: {e}")
        return []
    
def is_valid_url(url):
    parsed = urlparse(url)
    return all([parsed.scheme, parsed.netloc])

def load_website(url):
    headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        }

    if not is_valid_url(url):
        st.error("Invalid URL format.")
        return []
    
    try:
        docs = WebBaseLoader(url, header_template=headers).load()
        text_length = sum(len(doc.page_content) for doc in docs)

        if text_length < 800:  # Too short → probably JS site
            st.info("Detected JS-heavy site, switching to Selenium...")
            loader = SeleniumURLLoader(urls=[url] , browser='chrome')
            docs = loader.load()
            return docs

        st.write("Static site loaded with WebBaseLoader.")
        return docs

    except Exception as e:
        st.warning(f"WebBaseLoader failed: {e}")
        try:
            loader = SeleniumURLLoader(urls=[url] , browser='chrome')
            docs = loader.load()
            return docs
        
        except Exception as se:
            st.error(f"Selenium loader failed: {se}")
            return []

# ---------------------------------------------------
# CACHED VECTORSTORE BUILDER
# ---------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_vectorstore_from_documents(content_hash, documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    docs = splitter.split_documents(documents)

    embeddings = load_embeddings()
    # Use try-except for FAISS initialization
    try:
        return FAISS.from_documents(docs, embeddings)
    except Exception as e:
        st.error(f"Failed to create Vectorstore: {e}")
        st.stop()

# ---------------------------------------------------
# INITIALIZE CHATBOT
# ---------------------------------------------------
if initialize:

    st.session_state.messages = []
    st.session_state.chat_history = ChatMessageHistory()

    with st.spinner("Processing..."):

        # -----------------------------
        # LOAD DOCUMENTS
        # -----------------------------
        if user_options == "Document":
            if not uploaded_file:
                st.sidebar.error("Upload a file first.")
                st.stop()

            documents = []
            for file in uploaded_file:
                docs = load_document(file)
                documents.extend(docs)

        elif user_options == "Webpage":
            if not url:
                st.sidebar.error("Enter a URL.")
                st.stop()

            documents = load_website(url)

        if not documents:
            st.error("No content loaded.")
            st.stop()

        if not groq_api_key:
            st.error("GROQ_API_KEY not found in environment.")
            st.stop()

        # -----------------------------
        # VECTORSTORE
        # -----------------------------
        content_hash = hashlib.md5("".join([doc.page_content for doc in documents]).encode()).hexdigest()
        vectorstore = build_vectorstore_from_documents(content_hash, documents)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        # SAVE retriever
        st.session_state.retriever = retriever

        # -----------------------------
        # LLM
        # -----------------------------
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=1024,
            streaming=True
        )

        # -----------------------------
        # PROMPT
        # -----------------------------
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional AI research assistant.
             
            Use ONLY the provided context.
            If answer is not in context, say:
            "I could not find the answer in the provided document."

            Be clear and structured.
            """),
                        ("placeholder", "{history}"),
                        ("human", """
            Context:
            {context}

            Question:
            {question}
            """)
                    ])

        # -----------------------------
        # FORMAT DOCS
        # -----------------------------
        def format_docs(docs):
            return "\n\n".join(
                f"[Page {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
                for doc in docs
            )

        # -----------------------------
        # RAG CHAIN
        # -----------------------------
        rag_chain = (
            {
                "context": RunnableLambda(lambda x: x["question"]) 
                            | retriever 
                            | RunnableLambda(format_docs),
                "question": RunnableLambda(lambda x: x["question"]),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        conversational_chain = RunnableWithMessageHistory(
            rag_chain,
            lambda session_id: st.session_state.chat_history,
            input_messages_key="question",
            history_messages_key="history",
        )

        st.session_state.qa_chain = conversational_chain

    st.sidebar.success("✅ Chatbot Ready!")


# ---------------------------------------------------
# DISPLAY CHAT HISTORY
# ---------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------
if user_input := st.chat_input("Ask something about the uploaded content..."):

    if st.session_state.qa_chain is None:
        st.warning("Initialize chatbot from sidebar first.")
        st.stop()

    # Display user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                full_response = ""

                response_placeholder = st.empty()

                for chunk in st.session_state.qa_chain.stream(
                    {"question": user_input},
                    config={"configurable": {"session_id": "chat"}}
                ):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)

                answer = full_response

            except Exception as e:
                st.error(f"Error generating response: {e}")
                answer = "Sorry, I encountered an error while generating the response."
            
            source_docs = st.session_state.retriever.invoke(user_input)

            with st.expander("📚 View Source Chunks"):
                for i, doc in enumerate(source_docs):
                    st.markdown(f"**Source {i+1}:**")
                    st.write(doc.page_content[:500])
                    st.markdown("---")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )