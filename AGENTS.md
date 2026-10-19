# AGENTS.md

Notes for contributors and coding agents working on this repo.

## Install

```sh
python -m pip install -r requirements-dev.txt   # app requirements + pytest + playwright
python -m playwright install chromium           # add --with-deps on a fresh Linux host
```

## Test tiers

| Tier | What it covers | Secrets | When it runs |
| --- | --- | --- | --- |
| 1. Streamlit + Playwright smoke | `tests/test_streamlit_smoke.py` boots both `FTS/chat_with_pdf_search.py` and `GSI/chat_with_pdf_query.py` headlessly. For each app it checks that the title, welcome message, "Upload your PDF" form and chat input render (and, for GSI, the vector-index-ready message). It asks a question against the empty vector store, uploads a generated one-page PDF through the real LlamaIndex PDF reader and chunker, then asks again and checks that the RAG answer was built from the PDF's text. It also checks the `AUTH_ENABLED` password gate. Couchbase and OpenAI are replaced by in-memory fakes (`tests/fakes.py`, `tests/smoke_app.py`); the app code, Streamlit and LlamaIndex run for real. Every test fails on a `stException` element or on `Traceback` in the page or server log. | None | Every PR and push to `main` (`Streamlit + Playwright smoke (no secrets)` job). This is the required check. |
| 2. OpenAI provider smoke | `tests/test_openai_provider.py` calls the live OpenAI API through `llama_index.embeddings.openai.OpenAIEmbedding` and `llama_index.llms.openai.OpenAI`, with the exact models the apps use (read from both app files). Embeddings must be 1536 finite floats, matching both vector indexes, and the chat model must stream a non-empty reply through `SimpleChatEngine`. Couchbase is not used. | `OPENAI_API_KEY` | `OpenAI provider smoke (optional, OPENAI_API_KEY)` job, on PRs from this repo, pushes to `main`, and manual `workflow_dispatch`. If the secret is not available (not configured, fork or Dependabot PRs), the job passes with a "skipped" notice. |
| 3. Live Couchbase RAG validation | The full app against a real cluster, vector index and OpenAI. | All app secrets | Manual only (see below). |

Commands:

```sh
# Tier 1: no secrets needed (the tests strip any real credentials from the app's environment)
python -m pytest tests/test_streamlit_smoke.py -v

# Tier 2: skipped unless OPENAI_API_KEY is set in the environment
python -m pytest tests/test_openai_provider.py -v -rs
```

To run tier 2 on a Dependabot branch, start the `Smoke tests` workflow with
`workflow_dispatch` on that branch, or add `OPENAI_API_KEY` as a Dependabot
secret as well.

## Secrets

Names only. Never commit values. `.streamlit/secrets.toml` and `.env` are
git-ignored; keep it that way.

- `OPENAI_API_KEY`: optional GitHub Actions secret for tier 2. Also needed to run either app.
- `DB_CONN_STR`, `DB_USERNAME`, `DB_PASSWORD`, `DB_BUCKET`, `DB_SCOPE`,
  `DB_COLLECTION`: Couchbase settings for both apps, tier 3 only. They are not in CI.
- `INDEX_NAME`: the Search (FTS) vector index name, FTS app only.
- `AUTH_ENABLED`, `LOGIN_PASSWORD`: optional password gate for the UI.

## Manual live validation for dependency-update PRs

The smoke tests are guardrails. They do not replace a live end-to-end run.
Every dependency-update PR (including Dependabot PRs for `llama-index-*`,
`couchbase`, `streamlit`, `nltk`, etc.) needs a full manual validation against
real services before merge, with evidence on the PR. State whether you tested
the FTS path, the GSI path, or both (both is preferred; the GSI path needs
Couchbase Server 8.0+ or Capella).

1. **Dependency install**: `pip install -r requirements.txt` on the PR branch succeeds. Include the resolved versions of the bumped packages.
2. **Provider credentials**: name the secrets that were set (`OPENAI_API_KEY`, ...),
   never their values. Tier 2 passes, or explain why it was skipped.
3. **Couchbase connection**: the cluster at `DB_CONN_STR` is reachable and the
   `DB_BUCKET`/`DB_SCOPE`/`DB_COLLECTION` exist (the apps create the scope and
   collection if they are missing).
4. **App startup**: `streamlit run FTS/chat_with_pdf_search.py` and/or
   `streamlit run GSI/chat_with_pdf_query.py` start with no errors (paste the console output).
5. **Ingestion**: upload a small PDF in the sidebar. Note the "PDF loaded into
   vector store in N documents" message and the collection's document count.
6. **Vector index state**:
   - FTS: the Search index `INDEX_NAME` exists, is ready, and has a 1536-dim
     `embedding` vector field (`FTS/index.json`). Note its indexed document count.
   - GSI: `idx_vector_embedding` exists on the collection, its state in
     `system:indexes` is `online`, and the app shows "Vector index is ready".
7. **RAG query**: ask at least one question answered by the uploaded PDF.
   Record the question, the RAG answer (Couchbase logo) and the pure LLM answer
   (🤖️), and confirm that the RAG answer uses the PDF's content.
8. **Screenshots or traces** of the chat when practical.

If a live service is unavailable, write down the exact blocker on the PR. Do not
approve it based on the smoke tests alone.
