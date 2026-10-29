"""FastAPI-сервер: отдаёт три функции ассистента по HTTP.

Запуск: uvicorn api:app --port 8000
Документация: http://localhost:8000/docs
"""
import os
import tempfile

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from assistant import Assistant

AUDIO = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}

app = FastAPI(title="Ассистент по лекциям")
_assistant = Assistant()


def get_assistant():
    return _assistant


class TextIn(BaseModel):
    text: str


class AskIn(BaseModel):
    text: str
    question: str


def require(value, what):
    if not value.strip():
        raise HTTPException(400, f"Пустое поле: {what}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transcribe")
def transcribe(file: UploadFile = File(...), a: Assistant = Depends(get_assistant)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in AUDIO:
        raise HTTPException(400, f"Нужен аудиофайл: {', '.join(sorted(AUDIO))}")
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(file.file.read())
    try:
        text = a.transcribe(tmp.name)
    finally:
        os.remove(tmp.name)
    return {"text": text, "chars": len(text)}


@app.post("/summary")
def summary(body: TextIn, a: Assistant = Depends(get_assistant)):
    require(body.text, "text")
    return {"summary": a.summary(body.text)}


@app.post("/ask")
def ask(body: AskIn, a: Assistant = Depends(get_assistant)):
    require(body.text, "text")
    require(body.question, "question")
    answer, fragments = a.ask(body.text, body.question)
    return {"answer": answer, "fragments": fragments}
