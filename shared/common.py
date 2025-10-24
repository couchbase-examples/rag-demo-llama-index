"""
Common functions shared between both vector store implementations.
"""
import os
import tempfile
import streamlit as st
from datetime import timedelta

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.core.chat_engine.simple import SimpleChatEngine
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


def store_document(uploaded_file, storage_context):
    """Chunk the PDF & store it in Couchbase Vector Store."""
    if uploaded_file is not None:
        temp_dir = tempfile.TemporaryDirectory()
        temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)

        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        loader = SimpleDirectoryReader(input_files=[temp_file_path])
        documents = loader.load_data()

        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
        )
        st.info(f"PDF loaded into vector store in {len(documents)} documents")
        return index
    return None


@st.cache_resource(show_spinner="Connecting to Couchbase")
def connect_to_couchbase(connection_string, db_username, db_password):
    """Connect to couchbase"""
    from couchbase.cluster import Cluster
    from couchbase.auth import PasswordAuthenticator
    from couchbase.options import ClusterOptions

    auth = PasswordAuthenticator(db_username, db_password)
    options = ClusterOptions(auth)
    connect_string = connection_string
    cluster = Cluster(connect_string, options)

    # Wait until the cluster is ready for use.
    cluster.wait_until_ready(timedelta(seconds=5))

    return cluster


def setup_llm_and_embeddings():
    """Setup OpenAI LLM and embeddings"""
    llm = OpenAI(temperature=0, model="gpt-4o-mini")
    embeddings = OpenAIEmbedding(model='text-embedding-3-small')
    
    # Set the global settings for loading documents
    Settings.embed_model = embeddings
    Settings.chunk_size = 1500
    Settings.chunk_overlap = 150
    
    return llm, embeddings


def create_pure_llm_chat_engine(template_without_rag):
    """Create a pure LLM chat engine for comparison"""
    pure_llm = OpenAI(model="gpt-4o-mini")
    return SimpleChatEngine.from_defaults(
        llm=pure_llm,
        system_prompt=template_without_rag,
    )


def setup_sidebar_content():
    """Setup the sidebar content"""
    st.subheader("How does it work?")
    st.markdown(
        """
        For each question, you will get two answers:
        * one using RAG ([Couchbase logo](https://emoji.slack-edge.com/T024FJS4M/couchbase/4a361e948b15ed91.png))
        * one using pure LLM - OpenAI (🤖️).
        """
    )

    st.markdown(
        "For RAG, we are using [LlamaIndex](https://www.llamaindex.ai/), [Couchbase Vector Search](https://couchbase.com/) & [OpenAI](https://openai.com/). We fetch parts of the PDF relevant to the question using Vector search & add it as the context to the LLM. The LLM is instructed to answer based on the context from the Vector Store."
    )

    # View Code
    if st.checkbox("View Code"):
        st.write(
            "View the code here: [Github](https://github.com/couchbase-examples/rag-demo-llama-index/blob/main/chat_with_pdf.py)"
        )


def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "Hi, I'm a chatbot who can chat with the PDF. How can I help you?",
                "avatar": "🤖️",
            }
        )
        st.session_state.chat_llm = None
        st.session_state.chat_engine_rag = None


def handle_chat_interaction(couchbase_logo):
    """Handle the main chat interaction logic"""
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"])

    # React to user input
    if question := st.chat_input("Ask a question based on the PDF"):
        # Display user message in chat message container
        st.chat_message("user").markdown(question)

        # Add user message to chat history
        st.session_state.messages.append(
            {"role": "user", "content": question, "avatar": "👤"}
        )

        # Add placeholder for streaming the response
        with st.chat_message("assistant", avatar=couchbase_logo):
            message_placeholder = st.empty()

        # stream the response from the RAG
        rag_response = ""
        rag_stream_response = st.session_state.chat_engine_rag.stream_chat(question)
        for chunk in rag_stream_response.response_gen:
            rag_response += chunk
            message_placeholder.markdown(rag_response + "▌")

        message_placeholder.markdown(rag_response)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": rag_response,
                "avatar": couchbase_logo,
            }
        )

        # stream the response from the pure LLM

        # Add placeholder for streaming the response
        with st.chat_message("ai", avatar="🤖️"):
            message_placeholder_pure_llm = st.empty()

        pure_llm_response = ""
        pure_llm_stream_response = st.session_state.chat_llm.stream_chat(question)

        for chunk in pure_llm_stream_response.response_gen:
            pure_llm_response += chunk
            message_placeholder_pure_llm.markdown(pure_llm_response + "▌")

        message_placeholder_pure_llm.markdown(pure_llm_response)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": pure_llm_response,
                "avatar": "🤖️",
            }
        )