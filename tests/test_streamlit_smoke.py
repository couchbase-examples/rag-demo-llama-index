"""No-secret smoke test: boot each Streamlit app and drive it with Playwright."""

from dataclasses import dataclass

import pytest
from playwright.sync_api import Page, expect, sync_playwright

from fakes import PDF_MARKER

ANSWER = "Smoke answer from gpt-4o-mini"
RAG_ANSWER = f"{ANSWER} (with PDF context)"
# smoke_app.py sets every secret, including LOGIN_PASSWORD, to this placeholder.
LOGIN_PASSWORD = "smoke-placeholder"

# The first run of a fresh server imports LlamaIndex, which can take a while.
expect.set_options(timeout=30_000)


@dataclass
class App:
    path: str
    title: str
    welcome: str
    chat_placeholder: str
    # Text shown at startup and after an upload; only the GSI app has these.
    ready_text: str = ""
    uploaded_text: str = ""


APPS = [
    App(
        path="FTS/chat_with_pdf_search.py",
        title="Chat with PDF",
        welcome="Hi, I'm a chatbot who can search through existing documents in Couchbase",
        chat_placeholder="Ask a question - I'll search existing documents in Couchbase",
    ),
    App(
        path="GSI/chat_with_pdf_query.py",
        title="Chat with Database & PDF (QueryVectorStore)",
        welcome="Hi! I'm a chatbot that can search your database and answer questions.",
        chat_placeholder="Ask a question",
        ready_text="Vector index is ready for optimized search!",
        uploaded_text="Vector index updated successfully!",
    ),
]


def _pdf(text):
    """A one-page PDF with a single line of text."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        yield browser
        browser.close()


def _open(browser, url):
    page = browser.new_page()
    page.set_default_timeout(30_000)
    page.goto(url)
    return page


def _assert_no_errors(page: Page, log_path):
    expect(page.get_by_test_id("stException")).to_have_count(0)
    assert "Traceback" not in page.inner_text("body")
    log = log_path.read_text()
    assert "Traceback" not in log, log


def _assert_chat_ui(page: Page, app: App):
    expect(page.get_by_role("heading", name=app.title, exact=True)).to_be_visible()
    expect(page.get_by_text(app.welcome)).to_be_visible()
    expect(page.get_by_role("heading", name="Upload your PDF")).to_be_visible()
    expect(page.get_by_test_id("stFileUploader")).to_be_visible()
    expect(page.get_by_test_id("stFormSubmitButton").get_by_role("button", name="Upload")).to_be_visible()
    expect(page.get_by_placeholder(app.chat_placeholder, exact=True)).to_be_visible()
    if app.ready_text:
        expect(page.get_by_text(app.ready_text)).to_be_visible()


def _ask(page: Page, app: App, question):
    chat = page.get_by_placeholder(app.chat_placeholder, exact=True)
    chat.fill(question)
    chat.press("Enter")
    expect(page.get_by_text(question, exact=True)).to_be_visible()


@pytest.fixture(scope="module", params=APPS, ids=lambda app: app.path.split("/")[0])
def session(request, start_streamlit, browser):
    app = request.param
    url, log_path = start_streamlit(app.path, AUTH_ENABLED="False")
    page = _open(browser, url)
    yield app, page, log_path
    page.close()


def test_app_renders(session):
    app, page, log_path = session
    _assert_chat_ui(page, app)
    _assert_no_errors(page, log_path)


def test_chat_before_upload(session):
    app, page, log_path = session
    _ask(page, app, "What is Couchbase?")
    # The vector store is still empty, so LlamaIndex's RAG engine has no
    # context to answer from; the pure LLM answers as usual.
    expect(page.get_by_text("Empty Response", exact=True)).to_have_count(1)
    expect(page.get_by_text(ANSWER, exact=True)).to_have_count(1)
    _assert_no_errors(page, log_path)


def test_pdf_upload_and_rag_answer(session, tmp_path):
    app, page, log_path = session
    pdf = tmp_path / "smoke.pdf"
    pdf.write_bytes(_pdf(f"{PDF_MARKER}: Couchbase supports vector search."))
    page.get_by_test_id("stFileUploader").locator("input[type=file]").set_input_files(pdf)
    expect(page.get_by_text("smoke.pdf")).to_be_visible()
    page.get_by_test_id("stFormSubmitButton").get_by_role("button", name="Upload").click()
    expect(page.get_by_text("PDF loaded into vector store in 1 documents")).to_be_visible()
    if app.uploaded_text:
        expect(page.get_by_text(app.uploaded_text)).to_be_visible()
    _assert_no_errors(page, log_path)

    _ask(page, app, "What does Couchbase support?")
    # The RAG engine retrieved the uploaded PDF; the pure LLM answer did not.
    expect(page.get_by_text(RAG_ANSWER, exact=True)).to_have_count(1)
    expect(page.get_by_text(ANSWER, exact=True)).to_have_count(2)
    _assert_no_errors(page, log_path)


@pytest.mark.parametrize("app", APPS, ids=lambda app: app.path.split("/")[0])
def test_password_gate(app, start_streamlit, browser):
    url, log_path = start_streamlit(app.path, AUTH_ENABLED="True")
    page = _open(browser, url)
    password = page.get_by_label("Enter password")
    expect(password).to_be_visible()
    expect(page.get_by_placeholder(app.chat_placeholder, exact=True)).to_have_count(0)

    password.fill("wrong-password")
    page.get_by_role("button", name="Submit").click()
    expect(page.get_by_text("Incorrect password")).to_be_visible()
    expect(page.get_by_placeholder(app.chat_placeholder, exact=True)).to_have_count(0)

    password.fill(LOGIN_PASSWORD)
    page.get_by_role("button", name="Submit").click()
    _assert_chat_ui(page, app)
    _assert_no_errors(page, log_path)
    page.close()
