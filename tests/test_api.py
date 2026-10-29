import pytest
from fastapi.testclient import TestClient

from api import app, get_assistant
from assistant import Assistant
from fakes import FakeASR, FakeEmbedder, FakeLLM

TEXT = "Мозг растёт до трёх лет. Во сне мозг закрепляет память. С возрастом нейронов меньше."


@pytest.fixture
def client():
    fake = Assistant(asr=FakeASR("распознанный текст"), llm=FakeLLM("ответ"),
                     embedder=FakeEmbedder())
    app.dependency_overrides[get_assistant] = lambda: fake
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_transcribe(client):
    r = client.post("/transcribe", files={"file": ("lecture.mp3", b"ID3fake", "audio/mpeg")})
    assert r.status_code == 200
    assert r.json() == {"text": "распознанный текст", "chars": 18}


def test_transcribe_rejects_not_audio(client):
    r = client.post("/transcribe", files={"file": ("notes.pdf", b"%PDF", "application/pdf")})
    assert r.status_code == 400


def test_summary(client):
    r = client.post("/summary", json={"text": TEXT})
    assert r.status_code == 200
    assert r.json() == {"summary": "ответ"}


def test_summary_empty_text(client):
    assert client.post("/summary", json={"text": "  "}).status_code == 400


def test_summary_missing_field(client):
    assert client.post("/summary", json={}).status_code == 422


def test_ask(client):
    r = client.post("/ask", json={"text": TEXT, "question": "что делает мозг во сне?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "ответ"
    assert 1 <= len(body["fragments"]) <= 3
    assert {"text", "score"} <= body["fragments"][0].keys()


@pytest.mark.parametrize("payload", [
    {"text": "", "question": "вопрос"},
    {"text": TEXT, "question": "   "},
])
def test_ask_empty_fields(client, payload):
    assert client.post("/ask", json=payload).status_code == 400
