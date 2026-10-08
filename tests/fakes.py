"""In-memory stand-ins for Couchbase and OpenAI used by the no-secret smoke tests.

Only the external services are faked. The app code, Streamlit, LlamaIndex
and the PDF reader run for real.
"""

from typing import Any, List

import couchbase.cluster
import llama_index.embeddings.openai
import llama_index.llms.openai
import llama_index.vector_stores.couchbase
from llama_index.core.base.llms.types import (
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import CustomLLM
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from pydantic import PrivateAttr

# The smoke tests upload a PDF containing this phrase, and the fake LLM
# reports whether it reached the prompt, which proves retrieval worked.
PDF_MARKER = "Couchbase smoke marker"
SECRET_NAMES = (
    "OPENAI_API_KEY",
    "DB_CONN_STR",
    "DB_USERNAME",
    "DB_PASSWORD",
    "DB_BUCKET",
    "DB_SCOPE",
    "DB_COLLECTION",
    "INDEX_NAME",
    "LOGIN_PASSWORD",
)


class FakeQueryResult:
    def __init__(self, rows=()):
        self._rows = list(rows)

    def rows(self):
        return iter(self._rows)

    def execute(self, *args, **kwargs):
        return self._rows


class FakeCollectionManager:
    def create_scope(self, scope_name):
        pass

    def create_collection(self, scope_name, collection_name, *args, **kwargs):
        pass


class FakeSearchIndexManager:
    def upsert_index(self, index):
        pass


class FakeScope:
    def search_indexes(self):
        return FakeSearchIndexManager()

    def query(self, statement, *args, **kwargs):
        return FakeQueryResult()


class FakeBucket:
    def collections(self):
        return FakeCollectionManager()

    def scope(self, name):
        return FakeScope()


class FakeCluster:
    def __init__(self, connection_string, options):
        pass

    def wait_until_ready(self, timeout):
        pass

    def bucket(self, name):
        return FakeBucket()

    def query(self, statement, *args, **kwargs):
        # The GSI app checks system:indexes for its vector index.
        return FakeQueryResult([{"name": "idx_vector_embedding", "state": "online"}])


class FakeVectorStore(BasePydanticVectorStore):
    """Keeps nodes in memory and ranks them by dot product."""

    stores_text: bool = True
    flat_metadata: bool = False
    _nodes: dict = PrivateAttr(default_factory=dict)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

    @property
    def client(self) -> Any:
        return None

    def add(self, nodes: List[Any], **kwargs: Any) -> List[str]:
        for node in nodes:
            self._nodes[node.node_id] = node
        return [node.node_id for node in nodes]

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        self._nodes = {
            k: n for k, n in self._nodes.items() if n.ref_doc_id != ref_doc_id
        }

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        scored = sorted(
            (
                (sum(a * b for a, b in zip(query.query_embedding, n.embedding)), n)
                for n in self._nodes.values()
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )[: query.similarity_top_k]
        return VectorStoreQueryResult(
            nodes=[TextNode(text=n.get_content(), id_=n.node_id) for _, n in scored],
            similarities=[s for s, _ in scored],
            ids=[n.node_id for _, n in scored],
        )


class FakeOpenAI(CustomLLM):
    """Answers every prompt with a fixed reply naming the model."""

    model: str = "gpt-4o-mini"

    def __init__(self, model: str = "gpt-4o-mini", **kwargs: Any) -> None:
        super().__init__(model=model)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=self.model)

    def _answer(self, prompt: str) -> str:
        answer = f"Smoke answer from {self.model}"
        if PDF_MARKER in prompt:
            answer += " (with PDF context)"
        return answer

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(text=self._answer(prompt))

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        text = ""
        for word in self._answer(prompt).split(" "):
            delta = word if not text else " " + word
            text += delta
            yield CompletionResponse(text=text, delta=delta)


def fake_embedding(model: str = "text-embedding-3-small", **kwargs: Any) -> MockEmbedding:
    return MockEmbedding(embed_dim=1536)


def install(setattr=setattr):
    """Swap the external services for fakes. Pass monkeypatch.setattr to undo later."""
    setattr(couchbase.cluster, "Cluster", FakeCluster)
    setattr(llama_index.vector_stores.couchbase, "CouchbaseSearchVectorStore", FakeVectorStore)
    setattr(llama_index.vector_stores.couchbase, "CouchbaseQueryVectorStore", FakeVectorStore)
    setattr(llama_index.llms.openai, "OpenAI", FakeOpenAI)
    setattr(llama_index.embeddings.openai, "OpenAIEmbedding", fake_embedding)
