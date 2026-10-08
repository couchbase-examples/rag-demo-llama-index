"""Optional live OpenAI smoke test. Needs OPENAI_API_KEY; Couchbase is not used.

The model names are read from both apps, so this checks the exact models
they call through the same LlamaIndex OpenAI integrations.
"""

import ast
import math
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
APPS = ("FTS/chat_with_pdf_search.py", "GSI/chat_with_pdf_query.py")

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY is not set; skipping live OpenAI provider smoke test",
)


def _models(class_name):
    """The model= string literals passed to class_name(...) in either app."""
    models = {
        kw.value.value
        for app in APPS
        for call in ast.walk(ast.parse((REPO / app).read_text()))
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == class_name
        for kw in call.keywords
        if kw.arg == "model" and isinstance(kw.value, ast.Constant)
    }
    assert models, f"no {class_name}(model=...) call found in {APPS}"
    return sorted(models)


@pytest.mark.parametrize("model", _models("OpenAIEmbedding"))
def test_embedding(model):
    from llama_index.embeddings.openai import OpenAIEmbedding

    vector = OpenAIEmbedding(model=model).get_query_embedding("Couchbase")
    # FTS/index.json and the GSI vector index both expect 1536 dimensions.
    assert len(vector) == 1536
    assert all(isinstance(x, float) and math.isfinite(x) for x in vector)


@pytest.mark.parametrize("model", _models("OpenAI"))
def test_chat_model(model):
    from llama_index.core.chat_engine.simple import SimpleChatEngine
    from llama_index.llms.openai import OpenAI

    # The apps stream answers through SimpleChatEngine / ContextChatEngine.
    engine = SimpleChatEngine.from_defaults(llm=OpenAI(model=model, temperature=0, max_retries=1))
    reply = "".join(engine.stream_chat("Reply with the single word: pong").response_gen)
    assert reply.strip()
