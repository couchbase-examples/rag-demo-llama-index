import os
import tempfile

import streamlit as st

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)

from llama_index.core.chat_engine.simple import SimpleChatEngine
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.couchbase import CouchbaseSearchVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from couchbase.management.collections import CollectionSpec
from couchbase.exceptions import (
    ScopeAlreadyExistsException,
    CollectionAlreadyExistsException,
)

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
    from datetime import timedelta

    auth = PasswordAuthenticator(db_username, db_password)
    options = ClusterOptions(auth)
    connect_string = connection_string
    cluster = Cluster(connect_string, options)

    # Wait until the cluster is ready for use.
    cluster.wait_until_ready(timedelta(seconds=5))

    return cluster


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


@st.cache_resource()
def get_vector_store(
    _cluster,
    db_bucket,
    db_scope,
    db_collection,
    index_name,
):
    """Return the Couchbase vector store."""
    return CouchbaseSearchVectorStore(
        cluster=_cluster,
        bucket_name=db_bucket,
        scope_name=db_scope,
        collection_name=db_collection,
        index_name=index_name,
    )


if __name__ == "__main__":
    # Authorization
    if "auth" not in st.session_state:
        st.session_state.auth = False

    st.set_page_config(
        page_title="Chat with your PDF using LlamaIndex, Couchbase & OpenAI",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="auto",
        menu_items=None,
    )

    AUTH_ENABLED = parse_bool(os.getenv("AUTH_ENABLED", "False"))

    if not AUTH_ENABLED:
        st.session_state.auth = True
    else:
        # Authorization
        if "auth" not in st.session_state:
            st.session_state.auth = False

        AUTH = os.getenv("LOGIN_PASSWORD")
        check_environment_variable("LOGIN_PASSWORD")

        # Authentication
        user_pwd = st.text_input("Enter password", type="password")
        pwd_submit = st.button("Submit")

        if pwd_submit and user_pwd == AUTH:
            st.session_state.auth = True
        elif pwd_submit and user_pwd != AUTH:
            st.error("Incorrect password")

    if st.session_state.auth:
        # Load environment variables
        DB_CONN_STR = os.getenv("DB_CONN_STR")
        DB_USERNAME = os.getenv("DB_USERNAME")
        DB_PASSWORD = os.getenv("DB_PASSWORD")
        DB_BUCKET = os.getenv("DB_BUCKET")
        DB_SCOPE = os.getenv("DB_SCOPE")
        DB_COLLECTION = os.getenv("DB_COLLECTION")
        INDEX_NAME = os.getenv("INDEX_NAME")

        # Ensure that all environment variables are set
        check_environment_variable("OPENAI_API_KEY")
        check_environment_variable("DB_CONN_STR")
        check_environment_variable("DB_USERNAME")
        check_environment_variable("DB_PASSWORD")
        check_environment_variable("DB_BUCKET")
        check_environment_variable("DB_SCOPE")
        check_environment_variable("DB_COLLECTION")
        check_environment_variable("INDEX_NAME")

        # Connect to Couchbase Vector Store
        cluster = connect_to_couchbase(DB_CONN_STR, DB_USERNAME, DB_PASSWORD)

        # Ensure scope and collection exist
        ensure_scope_and_collection(cluster, DB_BUCKET, DB_SCOPE, DB_COLLECTION)

        vector_store = get_vector_store(
            cluster,
            DB_BUCKET,
            DB_SCOPE,
            DB_COLLECTION,
            INDEX_NAME,
        )

        # Build the prompt for the RAG
        template_rag = """You are a helpful bot. If you cannot answer based on the context provided, respond with a generic answer. Answer the question as truthfully as possible using the context below:
        {context}

        Question: {question}"""

        # Pure OpenAI prompt without RAG
        template_without_rag = """You are a helpful bot. Answer the question as truthfully as possible.

        Question: {question}"""

        # Frontend
        couchbase_logo = (
            "https://emoji.slack-edge.com/T024FJS4M/couchbase/4a361e948b15ed91.png"
        )

        st.title("Chat with PDF")
        st.markdown(
            "Answers with [Couchbase logo](https://emoji.slack-edge.com/T024FJS4M/couchbase/4a361e948b15ed91.png) are generated using *RAG* from existing documents in Couchbase while 🤖️ are generated by pure *LLM (ChatGPT)*"
        )

        # Use OpenAI as the llm & for embeddings
        llm = OpenAI(temperature=0, model="gpt-4o-mini")
        embeddings = OpenAIEmbedding(model='text-embedding-3-small')

        # Set the global settings for loading documents
        Settings.embed_model = embeddings
        Settings.chunk_size = 1500
        Settings.chunk_overlap = 150
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create a persistent RAG engine that can search existing documents in Couchbase
        existing_index = VectorStoreIndex.from_vector_store(vector_store)
        persistent_chat_engine_rag = existing_index.as_chat_engine(
            chat_mode="context",
            llm=llm,
            system_prompt=template_rag,
        )

        # Pure LLM for comparison of results
        pure_llm = OpenAI(model="gpt-4o-mini")
        st.session_state.chat_llm = SimpleChatEngine.from_defaults(
            llm=pure_llm,
            system_prompt=template_without_rag,
        )

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

                    # Create the chat engine with context from the uploaded data
                    st.session_state.chat_engine_rag = index.as_chat_engine(
                        chat_mode="context",
                        llm=llm,
                        system_prompt=template_rag,
                    )

            st.subheader("How does it work?")
            st.markdown(
                """
                For each question, you'll get two answers:
                * one using RAG from existing Couchbase documents ([Couchbase logo](https://emoji.slack-edge.com/T024FJS4M/couchbase/4a361e948b15ed91.png))
                * one using pure LLM - OpenAI (🤖️).
                
                You can upload new PDFs to add more content to the vector store!
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

        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Hi, I'm a chatbot who can search through existing documents in Couchbase and chat with you. You can also upload new PDFs to add more content. How can I help you?",
                    "avatar": "🤖️",
                }
            )
            st.session_state.chat_llm = None
            st.session_state.chat_engine_rag = None

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=message["avatar"]):
                st.markdown(message["content"])

        # React to user input
        if question := st.chat_input("Ask a question - I'll search existing documents in Couchbase"):
            # Display user message in chat message container
            st.chat_message("user").markdown(question)

            # Add user message to chat history
            st.session_state.messages.append(
                {"role": "user", "content": question, "avatar": "👤"}
            )

            # Always provide RAG response from existing Couchbase data
            with st.chat_message("assistant", avatar=couchbase_logo):
                message_placeholder = st.empty()

            # Use the session-specific RAG engine if available, otherwise use persistent one
            rag_engine = st.session_state.chat_engine_rag if st.session_state.chat_engine_rag is not None else persistent_chat_engine_rag
            
            try:
                # stream the response from the RAG
                rag_response = ""
                rag_stream_response = rag_engine.stream_chat(question)
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
                error_msg = f"Error searching Couchbase: {str(e)}"
                message_placeholder.markdown(error_msg)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_msg,
                        "avatar": couchbase_logo,
                    }
                )

            # Always provide pure LLM response
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