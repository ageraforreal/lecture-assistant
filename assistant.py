"""Логика ассистента: три модели и поиск по смыслу (RAG).

Модели загружаются только при первом обращении, поэтому модуль можно
импортировать без GPU, а в тестах подменить модели заглушками.
"""

WHISPER = "openai/whisper-small"                # речь -> текст
LLM = "Qwen/Qwen2.5-1.5B-Instruct"              # конспект и ответы
EMBEDDER = "intfloat/multilingual-e5-small"     # поиск по смыслу

SYSTEM = "Ты помощник студента. Отвечай по-русски."
CHUNK = 500          # размер куска лекции для поиска, символов
OVERLAP = 100        # перекрытие кусков, чтобы мысль не резалась на границе
PART = 6000          # столько символов Qwen получает за один запрос
TOP_K = 3            # сколько кусков передаётся в ответ на вопрос


def split_chunks(text, size=CHUNK, overlap=OVERLAP):
    """Режет текст на куски по size символов с перекрытием overlap."""
    text = text.strip()
    if not text:
        return []
    step = size - overlap
    return [text[i:i + size] for i in range(0, max(len(text) - overlap, 1), step)]


def dot(a, b):
    # векторы нормированы, поэтому скалярное произведение = косинусная близость
    return float(sum(x * y for x, y in zip(a, b)))


class Assistant:
    def __init__(self, asr=None, llm=None, embedder=None):
        self._asr, self._llm, self._embedder = asr, llm, embedder
        self._indexed_text, self._chunks, self._vecs = None, [], []

    # --- загрузка моделей по требованию ---

    @property
    def asr(self):
        if self._asr is None:
            import torch
            from transformers import pipeline
            device = 0 if torch.cuda.is_available() else -1
            self._asr = pipeline("automatic-speech-recognition", model=WHISPER, device=device)
        return self._asr

    @property
    def llm(self):
        if self._llm is None:
            from transformers import pipeline
            self._llm = pipeline("text-generation", model=LLM, torch_dtype="auto", device_map="auto")
        return self._llm

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(EMBEDDER)
        return self._embedder

    # --- модель 1: Whisper ---

    def transcribe(self, path):
        out = self.asr(path, chunk_length_s=30, generate_kwargs={"language": "russian"})
        return out["text"].strip()

    # --- модель 2: Qwen ---

    def chat(self, prompt, n=500):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt}]
        return self.llm(msgs, max_new_tokens=n)[0]["generated_text"][-1]["content"]

    def summary(self, text):
        """Конспект всей лекции: длинный текст делится на части,
        по каждой делается конспект, затем они сводятся в один."""
        parts = [text[i:i + PART] for i in range(0, len(text), PART)]
        if len(parts) == 1:
            return self.chat(f"Сделай краткий конспект лекции пунктами:\n\n{text}")
        notes = [self.chat(f"Выпиши главные мысли этой части лекции пунктами:\n\n{p}", 300)
                 for p in parts]
        joined = "\n\n".join(notes)
        return self.chat(f"Объедини заметки по частям лекции в один краткий конспект пунктами, "
                         f"без повторов:\n\n{joined}")

    # --- модель 3: E5 + ответ на вопрос ---

    def index(self, text):
        if text == self._indexed_text:
            return
        self._chunks = split_chunks(text)
        self._vecs = self.embedder.encode(["passage: " + c for c in self._chunks],
                                          normalize_embeddings=True)
        self._indexed_text = text

    def search(self, text, question, k=TOP_K):
        self.index(text)
        q = self.embedder.encode("query: " + question, normalize_embeddings=True)
        scores = [dot(v, q) for v in self._vecs]
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [{"text": self._chunks[i], "score": round(scores[i], 3)} for i in best]

    def ask(self, text, question):
        found = self.search(text, question)
        ctx = "\n---\n".join(f["text"] for f in found)
        answer = self.chat("Ответь на вопрос только по фрагментам лекции. "
                           "Если во фрагментах нет ответа, так и напиши: "
                           "«В лекции об этом не говорится».\n\n"
                           f"Фрагменты:\n{ctx}\n\nВопрос: {question}", 300)
        return answer, found
