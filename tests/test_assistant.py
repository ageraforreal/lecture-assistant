from assistant import CHUNK, PART, Assistant, split_chunks
from fakes import FakeASR, FakeEmbedder, FakeLLM


def make(**kw):
    return Assistant(asr=kw.get("asr", FakeASR()), llm=kw.get("llm", FakeLLM()),
                     embedder=kw.get("embedder", FakeEmbedder()))


# --- нарезка текста ---

def test_split_empty():
    assert split_chunks("") == []
    assert split_chunks("   ") == []


def test_split_short_text_is_one_chunk():
    assert split_chunks("короткий текст") == ["короткий текст"]


def test_split_covers_whole_text_with_overlap():
    text = "".join(str(i % 10) for i in range(1234))
    chunks = split_chunks(text, size=500, overlap=100)
    assert all(len(c) <= 500 for c in chunks)
    assert chunks[0] == text[:500]
    assert chunks[1].startswith(text[400:500])      # перекрытие 100 символов
    assert chunks[-1].endswith(text[-50:])          # конец текста не потерян


def test_split_default_size():
    chunks = split_chunks("а" * 5000)
    assert max(len(c) for c in chunks) == CHUNK


# --- распознавание ---

def test_transcribe_passes_russian_and_strips():
    asr = FakeASR("Текст лекции")
    assert make(asr=asr).transcribe("lecture.mp3") == "Текст лекции"
    path, kwargs = asr.calls[0]
    assert path == "lecture.mp3"
    assert kwargs["generate_kwargs"]["language"] == "russian"
    assert kwargs["chunk_length_s"] == 30


# --- конспект ---

def test_summary_short_text_one_call():
    llm = FakeLLM("конспект")
    assert make(llm=llm).summary("короткая лекция") == "конспект"
    assert len(llm.prompts) == 1
    assert "короткая лекция" in llm.prompts[0]


def test_summary_long_text_uses_whole_lecture():
    llm = FakeLLM("заметка")
    text = "А" * PART + "Б" * PART + "КОНЕЦ"
    make(llm=llm).summary(text)
    assert len(llm.prompts) == 4                    # 3 части + итоговое объединение
    assert "КОНЕЦ" in llm.prompts[2]                # хвост лекции не отброшен
    assert "Объедини" in llm.prompts[-1]


# --- поиск и ответ ---

LECTURE = ("Вступление про курс и организационные вопросы. " * 15 +
           "Во сне мозг переносит память из гиппокампа в кору, сон нужен для памяти. " * 3 +
           "С возрастом в мозге становится меньше воды, это признак старения. " * 3)


def test_search_finds_relevant_fragment():
    found = make().search(LECTURE, "зачем нужен сон для памяти?")
    assert len(found) == 3
    assert "сон" in found[0]["text"]
    assert found[0]["score"] >= found[1]["score"] >= found[2]["score"]


def test_index_is_cached_for_same_text():
    emb = FakeEmbedder()
    a = make(embedder=emb)
    a.search(LECTURE, "сон")
    a.search(LECTURE, "старение")
    assert emb.calls == 3       # 1 раз куски лекции + 2 вопроса


def test_ask_puts_fragments_into_prompt():
    llm = FakeLLM("Сон переносит память в кору.")
    answer, found = make(llm=llm).ask(LECTURE, "зачем нужен сон?")
    assert answer == "Сон переносит память в кору."
    prompt = llm.prompts[-1]
    assert found[0]["text"] in prompt
    assert "зачем нужен сон?" in prompt
    assert "В лекции об этом не говорится" in prompt   # защита от выдумок
