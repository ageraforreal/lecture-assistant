"""Streamlit-интерфейс. Обращается к FastAPI-серверу по HTTP.

Запуск: streamlit run app.py
"""
import os

import requests
import streamlit as st

API = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Ассистент по лекциям", page_icon="🎓")
st.title("🎓 Ассистент по лекциям")
st.caption("Whisper — речь в текст · Qwen — конспект и ответы · E5 — поиск по смыслу")


def call(path, **kwargs):
    try:
        r = requests.post(f"{API}{path}", timeout=1800, **kwargs)
    except requests.ConnectionError:
        st.error(f"Сервер {API} недоступен. Запустите: uvicorn api:app --port 8000")
        st.stop()
    if r.status_code != 200:
        st.error(r.json().get("detail", r.text))
        st.stop()
    return r.json()


st.session_state.setdefault("text", "")

# 1. Текст лекции
st.header("1. Лекция")
audio = st.file_uploader("Аудиозапись лекции", type=["mp3", "wav", "m4a", "ogg", "flac"])
if audio and st.button("Распознать речь"):
    with st.spinner("Whisper распознаёт речь, это может занять несколько минут..."):
        res = call("/transcribe", files={"file": (audio.name, audio.getvalue())})
    st.session_state.text = res["text"]
    st.success(f"Готово: {res['chars']} символов")

st.session_state.text = st.text_area("Текст лекции (можно вставить свой)",
                                     st.session_state.text, height=200)
text = st.session_state.text

if not text.strip():
    st.info("Загрузите аудио или вставьте текст лекции.")
    st.stop()

# 2. Конспект
st.header("2. Конспект")
if st.button("Сделать конспект"):
    with st.spinner("Qwen пишет конспект..."):
        st.session_state.summary = call("/summary", json={"text": text})["summary"]
if "summary" in st.session_state:
    st.markdown(st.session_state.summary)
    st.download_button("Скачать конспект", st.session_state.summary, "konspekt.md")

# 3. Вопрос по лекции
st.header("3. Вопрос по лекции")
q = st.text_input("Ваш вопрос", placeholder="Например: что происходит с мозгом в пожилом возрасте?")
if q and st.button("Спросить"):
    with st.spinner("Ищу фрагменты и формулирую ответ..."):
        res = call("/ask", json={"text": text, "question": q})
    st.markdown(f"**Ответ:** {res['answer']}")
    with st.expander("Найденные фрагменты лекции"):
        for f in res["fragments"]:
            st.markdown(f"*близость {f['score']}*")
            st.write(f["text"])
