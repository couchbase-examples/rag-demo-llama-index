"""
Shared configuration and environment setup for both vector store implementations.
"""
import os
import streamlit as st


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
        'INDEX_NAME': os.getenv("INDEX_NAME"),
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
        "DB_COLLECTION",
        "INDEX_NAME"
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