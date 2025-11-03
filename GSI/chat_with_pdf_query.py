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
                "content": "Hi! I'm a chatbot that can search your database and answer questions. You can chat with me anytime, and upload a PDF for enhanced search capabilities. How can I help you?",
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
    if question := st.chat_input("Ask a question"):
        # Display user message in chat message container
        st.chat_message("user").markdown(question)

        # Add user message to chat history
        st.session_state.messages.append(
            {"role": "user", "content": question, "avatar": "👤"}
        )

        # Try RAG response first (if available)
        if st.session_state.chat_engine_rag is not None:
            try:
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
            except Exception as e:
                # If RAG fails, show error and continue with pure LLM
                st.warning("⚠️ RAG search failed, showing pure LLM response only.")

        # Always show pure LLM response for comparison
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


def create_vector_index(
    cluster,
    bucket_name: str,
    scope_name: str,
    collection_name: str,
    index_name: str = "idx_vector_embedding",
    embedding_key: str = "vector",
    dimension: int = 1536,
    similarity: str = "cosine",
    recreate: bool = False
):
    """Create a GSI vector index for better performance."""
    from couchbase.options import QueryOptions
    
    # Get the scope object for query execution
    bucket = cluster.bucket(bucket_name)
    scope = bucket.scope(scope_name)
    
    # Check if index already exists first
    try:
        # Prepare the check query
        check_query_string = "SELECT name FROM system:indexes WHERE name = $index_name"
        prepared_check = scope.query(check_query_string, prepared=True)
        
        # Execute with parameters
        result = prepared_check.execute(parameters={"index_name": index_name})
        existing = list(result.rows())
        
        if existing:
            if recreate:
                # Drop existing index first (DDL statements can't be fully parameterized)
                try:
                    drop_query_string = f"DROP INDEX `{collection_name}`.`{index_name}`"
                    prepared_drop = scope.query(drop_query_string, prepared=True)
                    prepared_drop.execute()
                    st.info(f"🗑️ Existing index '{index_name}' dropped")
                except Exception as drop_e:
                    st.warning(f"Could not drop existing index: {drop_e}")
            else:
                st.info(f"✅ Index '{index_name}' already exists")
                return True
    except Exception:
        pass  # Continue to create
    
    # Create the index (DDL statements have limited parameterization support)
    try:
        create_query_string = f"CREATE INDEX `{index_name}` ON `{collection_name}` ({embedding_key} VECTOR) USING GSI WITH {{\"dimension\": {dimension}, \"description\": \"IVF,SQ8\", \"similarity\": \"{similarity}\"}}"
        prepared_create = scope.query(create_query_string, prepared=True)
        prepared_create.execute()
        
        st.success(f"✅ Vector index '{index_name}' created successfully!")
        return True
        
    except Exception as e:
        error_msg = str(e)
        if "already exists" in error_msg.lower():
            st.info(f"✅ Index '{index_name}' already exists")
            return True
        else:
            st.error(f"❌ Failed to create index: {error_msg}")
            return False


def check_vector_index_exists(
    cluster,
    bucket_name: str,
    scope_name: str,
    collection_name: str,
    index_name: str = "idx_vector_embedding"
) -> bool:
    """Check if a vector index exists and is online."""
    try:
        from couchbase.options import QueryOptions
        
        # Use cluster-level query for better compatibility
        # More comprehensive query to find the index
        check_query = f"""
        SELECT name, state, keyspace_id, scope_id, bucket_id, index_key
        FROM system:indexes 
        WHERE name = '{index_name}'
        AND bucket_id = '{bucket_name}'
        """
        
        result = cluster.query(check_query, QueryOptions(timeout=timedelta(seconds=10)))
        rows = list(result.rows())
        
        if rows:
            index_info = rows[0]
            index_state = index_info.get('state', 'unknown')
            
            if index_state == 'online':
                st.success(f"✅ Vector index '{index_name}' is **online** and ready!")
                return True
            elif index_state in ['building', 'pending', 'deferred']:
                st.info(f"🔄 Vector index '{index_name}' is **{index_state}**")
                return False
            elif index_state == 'created':
                st.info(f"📋 Vector index '{index_name}' is **created** but not yet building")
                return False
            else:
                st.warning(f"⚠️ Vector index '{index_name}' state: **{index_state}**")
                return False
        else:
            return False
        
    except Exception as e:
        st.error(f"❌ Error checking vector index: {e}")
        return False


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
        search_type="ANN",
        similarity="cosine",
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
    """Create scope and collection if they do not already exist."""
    bucket = cluster.bucket(bucket_name)
    cm = bucket.collections()

    # Ensure scope exists
    try:
        cm.create_scope(scope_name)
    except ScopeAlreadyExistsException:
        pass
    except Exception as e:
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

    # Ensure scope and collection are present
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

    # Check vector index status
    index_exists = check_vector_index_exists(
        cluster,
        env_vars['DB_BUCKET'],
        env_vars['DB_SCOPE'], 
        env_vars['DB_COLLECTION']
    )
    
    if index_exists:
        st.success("✅ Vector index is ready for optimized search!")

    # Get prompt templates
    template_rag, template_without_rag = get_prompts()

    # Frontend
    couchbase_logo = "https://emoji.slack-edge.com/T024FJS4M/couchbase/4a361e948b15ed91.png"

    st.title("Chat with Database & PDF (QueryVectorStore)")
    st.markdown(
        "🔎 **QueryVectorStore Version** - Chat anytime! Answers with [Couchbase logo](https://emoji.slack-edge.com/T024FJS4M/couchbase/4a361e948b15ed91.png) use *RAG* (when data is available) while 🤖️ are pure *LLM (ChatGPT)* responses"
    )

    # Setup LLM and embeddings
    llm, embeddings = setup_llm_and_embeddings()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Pure LLM for comparison of results
    st.session_state.chat_llm = create_pure_llm_chat_engine(template_without_rag)
    
    # Initialize RAG chat engine with existing data if available
    if "chat_engine_rag" not in st.session_state or st.session_state.chat_engine_rag is None:
        if index_exists:
            try:
                # Try to create index from existing vector store data
                existing_index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    storage_context=storage_context
                )
                st.session_state.chat_engine_rag = existing_index.as_chat_engine(
                    chat_mode="context",
                    llm=llm,
                    system_prompt=template_rag,
                )
                st.success("✅ Connected to existing data in vector store!")
            except Exception:
                # No existing data, create empty index for future use
                st.info("ℹ️ No existing data found. You can still chat using pure LLM or upload a PDF for RAG!")
                # Create a placeholder index that will work with empty vector store
                try:
                    st.session_state.chat_engine_rag = VectorStoreIndex.from_vector_store(
                        vector_store=vector_store,
                        storage_context=storage_context
                    ).as_chat_engine(
                        chat_mode="context",
                        llm=llm,
                        system_prompt=template_rag,
                    )
                except Exception:
                    st.session_state.chat_engine_rag = None
        else:
            st.info("ℹ️ No vector index found. You can still chat using pure LLM or upload a PDF to enable RAG!")
            # Still allow chatting, but RAG won't work without index
            st.session_state.chat_engine_rag = None

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
                    
                    # Create vector index after new PDF upload
                    st.info("🔧 Creating vector index for new PDF data...")
                    with st.spinner("Updating vector index..."):
                        success = create_vector_index(
                            cluster,
                            env_vars['DB_BUCKET'],
                            env_vars['DB_SCOPE'],
                            env_vars['DB_COLLECTION'],
                            index_name="idx_vector_embedding",
                            embedding_key="vector",
                            dimension=1536,
                            similarity="cosine",
                            recreate=True  # Force recreation for new PDF
                        )
                        if success:
                            st.success("✅ Vector index updated successfully!")
                        else:
                            st.warning("⚠️ Vector index update failed, but search will still work")

        setup_sidebar_content()

    # Initialize session state
    initialize_session_state()

    # Always allow chat interaction
    handle_chat_interaction(couchbase_logo)


if __name__ == "__main__":
    main()
