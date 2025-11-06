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


def create_vector_index(
    cluster,
    bucket_name: str,
    scope_name: str,
    collection_name: str,
    index_name: str = "idx_vector_embedding",
    embedding_key: str = "vector",
    dimension: int = 1536,
    similarity: str = "cosine"
):
    """Create a GSI vector index for better performance."""
    import time
    from couchbase.options import QueryOptions
    
    st.info("🚀 **Starting Vector Index Creation Process**")
    
    # Step 1: Test basic permissions first
    try:
        test_query = "SELECT 1"
        cluster.query(test_query, QueryOptions(timeout=timedelta(seconds=5)))
        st.success("✅ Basic query permissions verified")
    except Exception as e:
        st.error(f"❌ Basic query failed: {e}")
        return False
    
    # Step 2: Check if we can access system:indexes
    try:
        sys_query = "SELECT name FROM system:indexes LIMIT 1"
        cluster.query(sys_query, QueryOptions(timeout=timedelta(seconds=5)))
        st.success("✅ Can access system:indexes")
    except Exception as e:
        st.error(f"❌ Cannot access system:indexes: {e}")
        st.error("**Fix**: Grant 'Query System Catalog' permission to your user")
        return False
    
    # Step 3: Simple check if index already exists
    try:
        check_query = f"SELECT name FROM system:indexes WHERE name = '{index_name}'"
        result = cluster.query(check_query, QueryOptions(timeout=timedelta(seconds=5)))
        existing = list(result.rows())
        if existing:
            st.warning(f"⚠️ Index '{index_name}' already exists. Let's check its status...")
            # Just return success if it exists - let the check function handle validation
            return True
    except Exception as e:
        st.warning(f"Could not check for existing index: {e}")
    
    # Step 4: Create the index with the simplest possible syntax
    try:
        # Use EXACT syntax that works in Capella Query Workbench with backticks
        create_query = f"CREATE INDEX `{index_name}` ON `{bucket_name}`.`{scope_name}`.`{collection_name}`(`{embedding_key}` VECTOR) WITH {{\"dimension\":{dimension},\"similarity\":\"{similarity}\"}}"
        
        st.info("📝 **Executing Index Creation Query:**")
        st.code(create_query, language="sql")
        
        # Execute with a reasonable timeout
        result = cluster.query(create_query, QueryOptions(timeout=timedelta(seconds=180)))
        st.success("✅ **Index creation command completed successfully!**")
        
        # Show query metrics if available
        try:
            metadata = result.metadata()
            if metadata and metadata.metrics():
                exec_time = metadata.metrics().execution_time
                st.info(f"⏱️ Query execution time: {exec_time}")
        except:
            pass
            
    except Exception as e:
        error_msg = str(e)
        st.error(f"❌ **Index creation failed**: {error_msg}")
        
        # Show full error details for debugging
        st.error("**Full Error Details:**")
        st.code(f"Error Type: {type(e).__name__}\nError Message: {error_msg}", language="text")
        
        # Detailed error analysis
        if "already exists" in error_msg.lower():
            st.info("💡 Index already exists - this is actually good!")
            return True
        elif "permission" in error_msg.lower() or "unauthorized" in error_msg.lower():
            st.error("🔐 **Permission Error**: Your user lacks index creation permissions")
            st.markdown("""
            **To Fix This:**
            1. Go to Capella UI → Access → Database Users
            2. Edit your user permissions  
            3. Add **'Query Manage Index'** permission for this bucket
            4. Try again
            """)
            return False
        elif "syntax" in error_msg.lower() or "parse" in error_msg.lower():
            st.error("📝 **SQL Syntax Error**: The CREATE INDEX statement has invalid syntax")
            return False
        elif "keyspace" in error_msg.lower() or "collection" in error_msg.lower():
            st.error("🗂️ **Collection Error**: The target collection may not exist")
            st.info("� Try uploading a PDF first to create documents in the collection")
            return False
        else:
            st.error(f"�🔍 **Unexpected Error**: {error_msg}")
            st.markdown("**Debug Steps:**")
            st.markdown("1. Try running this exact query in Capella Query Workbench:")
            st.code(create_query, language="sql")
            st.markdown("2. If it works there but not here, it's likely a permission or connection context issue")
            return False
    
    # Step 5: Wait for the index to become online (simplified)
    st.info("⏳ **Waiting for index to become online...**")
    
    max_attempts = 24  # 2 minutes with 5-second intervals
    attempt = 0
    
    while attempt < max_attempts:
        try:
            # Simple status check
            status_query = f"""
            SELECT name, state 
            FROM system:indexes 
            WHERE name = '{index_name}' 
            AND bucket_id = '{bucket_name}'
            """
            
            result = cluster.query(status_query, QueryOptions(timeout=timedelta(seconds=5)))
            rows = list(result.rows())
            
            if not rows:
                st.warning(f"🔍 Index not found yet... (attempt {attempt + 1}/{max_attempts})")
            else:
                index_state = rows[0].get('state', 'unknown')
                
                if index_state == 'online':
                    st.success(f"🎉 **SUCCESS!** Index '{index_name}' is now ONLINE!")
                    st.balloons()
                    return True
                elif index_state in ['building', 'pending', 'deferred', 'created']:
                    st.info(f"🔄 Index state: **{index_state}** (attempt {attempt + 1}/{max_attempts})")
                else:
                    st.warning(f"⚠️ Unexpected state: **{index_state}**")
            
            time.sleep(5)
            attempt += 1
            
        except Exception as e:
            st.warning(f"Status check failed: {e}")
            attempt += 1
            time.sleep(5)
    
    # If we reach here, the index creation timed out
    st.warning("⏰ **Timeout**: Index creation is taking longer than expected")
    st.info("💡 The index might still be building. Check Capella UI → Indexes section")
    st.info("💡 You can also try the 'Check Index Status' button later")
    
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
            keyspace = index_info.get('keyspace_id', '')
            scope_id = index_info.get('scope_id', '')
            bucket_id = index_info.get('bucket_id', '')
            index_key = index_info.get('index_key', '')
            
            st.info(f"""
            📋 **Index Details:**
            - **Name**: {index_name}
            - **State**: {index_state}
            - **Bucket**: {bucket_id}
            - **Scope**: {scope_id}  
            - **Collection**: {keyspace}
            - **Index Key**: {index_key}
            """)
            
            # Check if it's the right index for our collection
            is_correct_collection = (
                keyspace == collection_name and 
                scope_id == scope_name and 
                bucket_id == bucket_name
            )
            
            if not is_correct_collection:
                st.warning(f"Found index '{index_name}' but it's for a different collection")
                st.info(f"Expected: {bucket_name}.{scope_name}.{collection_name}")
                st.info(f"Found: {bucket_id}.{scope_id}.{keyspace}")
                return False
            
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
            # Let's also check all indexes to see what exists
            st.info(f"🔍 Index '{index_name}' not found. Checking all indexes in bucket...")
            
            all_indexes_query = f"""
            SELECT name, state, keyspace_id, scope_id, bucket_id, index_key
            FROM system:indexes 
            WHERE bucket_id = '{bucket_name}'
            ORDER BY name
            """
            all_result = cluster.query(all_indexes_query, QueryOptions(timeout=timedelta(seconds=10)))
            all_rows = list(all_result.rows())
            
            st.info(f"� **All indexes in bucket '{bucket_name}':**")
            if all_rows:
                vector_count = 0
                for idx in all_rows:
                    index_key_str = str(idx.get('index_key', ''))
                    is_vector = "VECTOR" in index_key_str.upper()
                    if is_vector:
                        vector_count += 1
                    
                    vector_indicator = "🔢" if is_vector else "📄"
                    scope_collection = f"{idx.get('scope_id', '')}.{idx.get('keyspace_id', '')}"
                    st.text(f"  {vector_indicator} **{idx.get('name')}** ({idx.get('state')}) - {scope_collection}")
                
                if vector_count == 0:
                    st.warning("⚠️ No vector indexes found in this bucket")
                else:
                    st.info(f"Found {vector_count} vector index(es) in bucket")
            else:
                st.warning("⚠️ No indexes found in this bucket")
            
            return False
        
    except Exception as e:
        st.error(f"❌ Error checking vector index: {e}")
        
        # Provide troubleshooting info
        error_msg = str(e).lower()
        if "permission" in error_msg or "unauthorized" in error_msg:
            st.error("**Permission Issue**: Your user needs permissions to query system:indexes")
        elif "timeout" in error_msg:
            st.warning("**Timeout Issue**: Query took too long to execute")
            
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
        similarity="cosine",  # Changed to cosine for better semantic similarity
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
                    
                    # Auto-create vector index after PDF upload
                    st.info("🔧 Creating vector index for better search performance...")
                    with st.spinner("Setting up vector index..."):
                        success = create_vector_index(
                            cluster,
                            env_vars['DB_BUCKET'],
                            env_vars['DB_SCOPE'],
                            env_vars['DB_COLLECTION'],
                            index_name="idx_vector_embedding",
                            embedding_key="vector",
                            dimension=1536,
                            similarity="cosine"
                        )
                        if success:
                            st.success("✅ Vector index created successfully!")
                        else:
                            st.warning("⚠️ Vector index creation failed, but search will still work")

        setup_sidebar_content()
        


            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔍 Check Index Status"):
                    index_exists = check_vector_index_exists(
                        cluster,
                        env_vars['DB_BUCKET'],
                        env_vars['DB_SCOPE'],
                        env_vars['DB_COLLECTION']
                    )
                    if index_exists:
                        st.success("✅ Vector index is active and ready!")
                    else:
                        st.warning("⚠️ No vector index found")
            
            with col2:
                if st.button("📋 List All Indexes"):
                    try:
                        from couchbase.options import QueryOptions
                        
                        query = f"""
                        SELECT name, state, keyspace_id, scope_id, bucket_id, index_key
                        FROM system:indexes 
                        WHERE bucket_id = '{env_vars['DB_BUCKET']}'
                        ORDER BY name
                        """
                        result = cluster.query(query, QueryOptions(timeout=timedelta(seconds=15)))
                        rows = list(result.rows())
                        
                        if rows:
                            st.info("📊 **All indexes in this bucket:**")
                            vector_count = 0
                            regular_count = 0
                            
                            for idx in rows:
                                index_key_str = str(idx.get('index_key', ''))
                                is_vector = "VECTOR" in index_key_str.upper()
                                
                                if is_vector:
                                    vector_count += 1
                                    vector_indicator = "🔢"
                                else:
                                    regular_count += 1
                                    vector_indicator = "📄"
                                
                                scope_collection = f"{idx.get('scope_id', '')}.{idx.get('keyspace_id', '')}"
                                state_emoji = "✅" if idx.get('state') == 'online' else "�" if idx.get('state') in ['building', 'pending'] else "⚠️"
                                
                                st.markdown(f"  {vector_indicator} **{idx.get('name')}** {state_emoji} `{idx.get('state')}` - *{scope_collection}*")
                            
                            st.info(f"📈 **Summary**: {vector_count} vector indexes, {regular_count} regular indexes")
                            
                            if vector_count == 0:
                                st.warning("⚠️ No vector indexes found - create one for better performance!")
                                
                        else:
                            st.warning("⚠️ No indexes found in this bucket")
                    except Exception as e:
                        st.error(f"❌ Error listing indexes: {e}")
                        if "permission" in str(e).lower():
                            st.error("**Permission Issue**: Need access to query system:indexes")
            
            st.divider()
            
            # Add a test connection button
            col3, col4 = st.columns(2)
            with col3:
                if st.button("🔧 Test Connection"):
                    try:
                        from couchbase.options import QueryOptions
                        
                        # Simple test query
                        test_query = "SELECT 'Connection OK' as status"
                        result = cluster.query(test_query, QueryOptions(timeout=timedelta(seconds=5)))
                        rows = list(result.rows())
                        if rows:
                            st.success("✅ Connection and basic query work!")
                        
                        # Test system:indexes access
                        sys_query = "SELECT COUNT(*) as count FROM system:indexes LIMIT 1"
                        sys_result = cluster.query(sys_query, QueryOptions(timeout=timedelta(seconds=5)))
                        sys_rows = list(sys_result.rows())
                        if sys_rows:
                            st.success("✅ Can access system:indexes!")
                        
                    except Exception as e:
                        st.error(f"❌ Connection test failed: {e}")
                        if "permission" in str(e).lower():
                            st.error("**Permission Issue**: Check user permissions")
            
            with col4:
                # Debug: Test the exact CREATE INDEX query
                if st.button("🔍 Debug Query", help="Test CREATE INDEX query syntax without actually creating"):
                    debug_query = f"CREATE INDEX `idx_vector_embedding` ON `{env_vars['DB_BUCKET']}`.`{env_vars['DB_SCOPE']}`.`{env_vars['DB_COLLECTION']}`(`vector` VECTOR) WITH {{\"dimension\":1536,\"similarity\":\"cosine\"}}"
                    st.code(debug_query, language="sql")
                    
                    try:
                        st.info("Testing query syntax and permissions...")
                        # We'll use EXPLAIN to test the syntax without actually creating
                        explain_query = f"EXPLAIN {debug_query}"
                        result = cluster.query(explain_query, QueryOptions(timeout=timedelta(seconds=10)))
                        rows = list(result.rows())
                        st.success("✅ Query syntax is valid and you have permissions!")
                        st.json(rows[0] if rows else {})
                    except Exception as e:
                        st.error(f"❌ Debug query failed: {e}")
                        st.code(f"Error Type: {type(e).__name__}\nMessage: {str(e)}", language="text")
                
                if st.button("🚀 Create Vector Index", type="primary"):
                    with st.spinner("Creating vector index..."):
                        success = create_vector_index(
                            cluster,
                            env_vars['DB_BUCKET'],
                            env_vars['DB_SCOPE'],
                            env_vars['DB_COLLECTION'],
                            index_name="idx_vector_embedding",
                            embedding_key="vector",
                            dimension=1536,
                            similarity="cosine"
                        )
                        if success:
                            st.balloons()
                            st.rerun()  # Refresh to show the new index status
            
            st.info(
                """
                **Vector Index Benefits:**
                - 🚀 Much faster search performance
                - 💾 Lower memory usage
                - 📈 Better scalability for large datasets
                - ⚡ Optimized ANN (Approximate Nearest Neighbor) search
                """
            )

    # Initialize session state
    initialize_session_state()

    # Handle chat interaction
    handle_chat_interaction(couchbase_logo)


if __name__ == "__main__":
    main()
