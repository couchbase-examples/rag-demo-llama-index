"""
Chat with PDF using LlamaIndex, Couchbase QueryVectorStore & OpenAI
This version uses CouchbaseQueryVectorStore for vector operations.
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
from llama_index.vector_stores.couchbase import CouchbaseQueryVectorStore
from couchbase.management.collections import CollectionSpec
from couchbase.exceptions import (
    ScopeAlreadyExistsException,
    CollectionAlreadyExistsException,
)


# Configuration functions
def parse_bool(value: str):
    """Parse boolean values from environment variables"""
    return value.lower() in ("yes", "true", "t", "1")


def check_environment_variable(variable_name):
    """Check if environment variable is set"""
    if variable_name not in os.environ:
        st.error(
            f"{variable_name} environment variable is not set. Please add it to the secrets.toml file"
        )
        st.stop()


def get_environment_variables():
    """Get all required environment variables"""
    return {
        'DB_CONN_STR': os.getenv("DB_CONN_STR"),
        'DB_USERNAME': os.getenv("DB_USERNAME"),
        'DB_PASSWORD': os.getenv("DB_PASSWORD"),
        'DB_BUCKET': os.getenv("DB_BUCKET"),
        'DB_SCOPE': os.getenv("DB_SCOPE"),
        'DB_COLLECTION': os.getenv("DB_COLLECTION"),
    }


def validate_environment_variables():
    """Ensure that all environment variables are set"""
    required_vars = [
        "OPENAI_API_KEY",
        "DB_CONN_STR", 
        "DB_USERNAME",
        "DB_PASSWORD",
        "DB_BUCKET",
        "DB_SCOPE", 
        "DB_COLLECTION"
    ]
    
    for var in required_vars:
        check_environment_variable(var)


def setup_authentication():
    """Handle authentication logic"""
    if "auth" not in st.session_state:
        st.session_state.auth = False

    AUTH_ENABLED = parse_bool(os.getenv("AUTH_ENABLED", "False"))

    if not AUTH_ENABLED:
        st.session_state.auth = True
        return True
    else:
        AUTH = os.getenv("LOGIN_PASSWORD")
        check_environment_variable("LOGIN_PASSWORD")

        # Authentication
        user_pwd = st.text_input("Enter password", type="password")
        pwd_submit = st.button("Submit")

        if pwd_submit and user_pwd == AUTH:
            st.session_state.auth = True
            return True
        elif pwd_submit and user_pwd != AUTH:
            st.error("Incorrect password")
            return False
        
        return st.session_state.auth


def get_prompts():
    """Get the prompt templates"""
    template_rag = """You are a helpful bot. If you cannot answer based on the context provided, respond with a generic answer. Answer the question as truthfully as possible using the context below:
    {context}

    Question: {question}"""

    template_without_rag = """You are a helpful bot. Answer the question as truthfully as possible.

    Question: {question}"""
    
    return template_rag, template_without_rag


# Common functions
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


def get_query_vector_store(
    _cluster,
    db_bucket,
    db_scope,
    db_collection,
):
    """Return the CouchbaseQueryVectorStore."""
    return CouchbaseQueryVectorStore(
        cluster=_cluster,
        bucket_name=db_bucket,
        scope_name=db_scope,
        collection_name=db_collection,
        search_type="KNN",
        similarity="euclidean",
        text_key="content",
        embedding_key="vector",
        metadata_key="meta",
    )


def ensure_scope_and_collection(
    cluster,
    bucket_name: str,
    scope_name: str,
    collection_name: str,
) -> None:
    """Create scope and collection if they do not already exist.

    Uses management API; safe to call repeatedly thanks to exception handling.
    """
    bucket = cluster.bucket(bucket_name)
    cm = bucket.collections()

    # Ensure scope exists
    try:
        cm.create_scope(scope_name)
    except ScopeAlreadyExistsException:
        pass
    except Exception as e:
        # If missing permissions ignore silently in demo context
        st.warning(f"Could not create scope '{scope_name}': {e}")

    # Ensure collection exists
    try:
        cm.create_collection(CollectionSpec(collection_name, scope_name))
    except CollectionAlreadyExistsException:
        pass
    except Exception as e:
        st.warning(f"Could not create collection '{collection_name}': {e}")


def main():
    """Main application function"""
    st.set_page_config(
        page_title="Chat with your PDF using LlamaIndex, Couchbase QueryVectorStore & OpenAI",
        page_icon="🔎",
        layout="centered",
        initial_sidebar_state="auto",
        menu_items=None,
    )

    # Authentication
    if not setup_authentication():
        return

    # Load and validate environment variables
    validate_environment_variables()
    env_vars = get_environment_variables()

    # Connect to Couchbase Vector Store
    cluster = connect_to_couchbase(
        env_vars['DB_CONN_STR'],
        env_vars['DB_USERNAME'],
        env_vars['DB_PASSWORD']
    )

    # Ensure scope and collection are present before index creation
    ensure_scope_and_collection(
        cluster,
        env_vars['DB_BUCKET'],
        env_vars['DB_SCOPE'],
        env_vars['DB_COLLECTION'],
    )

    vector_store = get_query_vector_store(
        cluster,
        env_vars['DB_BUCKET'],
        env_vars['DB_SCOPE'],
        env_vars['DB_COLLECTION'],
    )

    # Get prompt templates
    template_rag, template_without_rag = get_prompts()

    # Frontend
    couchbase_logo = "https://emoji.slack-edge.com/T024FJS4M/couchbase/4a361e948b15ed91.png"

    st.title("Chat with PDF (QueryVectorStore)")
    st.markdown(
        "🔎 **QueryVectorStore Version** - Answers with [Couchbase logo](https://emoji.slack-edge.com/T024FJS4M/couchbase/4a361e948b15ed91.png) are generated using *RAG* while 🤖️ are generated by pure *LLM (ChatGPT)*"
    )

    # Setup LLM and embeddings
    llm, embeddings = setup_llm_and_embeddings()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Pure LLM for comparison of results
    st.session_state.chat_llm = create_pure_llm_chat_engine(template_without_rag)

    with st.sidebar:
        st.header("Upload your PDF")
        with st.form("upload pdf"):
            uploaded_file = st.file_uploader(
                "Choose a PDF.",
                help="The document will be deleted after one hour of inactivity (TTL).",
                type="pdf",
            )
            submitted = st.form_submit_button("Upload")
            if submitted:
                index = store_document(uploaded_file, storage_context)
                if not index:
                    st.warning("Please upload a valid PDF")
                else:
                    # Create the chat engine with context from the uploaded data
                    st.session_state.chat_engine_rag = index.as_chat_engine(
                        chat_mode="context",
                        llm=llm,
                        system_prompt=template_rag,
                    )

        setup_sidebar_content()

    # Initialize session state
    initialize_session_state()

    # Handle chat interaction
    handle_chat_interaction(couchbase_logo)


if __name__ == "__main__":
    main()
