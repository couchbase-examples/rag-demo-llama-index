"""Run one of the Streamlit apps with Couchbase and OpenAI swapped for in-memory fakes.

Streamlit executes this file instead of the app so the UI can be smoke tested
without a Couchbase cluster or an OPENAI_API_KEY. SMOKE_APP picks the app,
relative to the repo root (for example FTS/chat_with_pdf_search.py).
"""

import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fakes  # noqa: E402

APP = Path(__file__).resolve().parent.parent / os.environ["SMOKE_APP"]

# Placeholders only, so the app's environment checks pass.
for name in fakes.SECRET_NAMES:
    os.environ[name] = "smoke-placeholder"

fakes.install()

runpy.run_path(str(APP), run_name="__main__")
