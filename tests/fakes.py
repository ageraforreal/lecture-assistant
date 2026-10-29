"""Заглушки моделей: ведут себя как настоящие, но работают мгновенно и без GPU."""
import math

WORDS = ["мозг", "нейрон", "старение", "память", "сон", "вода"]


class FakeASR:
    def __init__(self, text="Сегодня поговорим про мозг."):
        self.text, self.calls = text, []

    def __call__(self, path, **kwargs):
        self.calls.append((path, kwargs))
        return {"text": " " + self.text}


class FakeLLM:
    """Отвечает фиксированной строкой и запоминает, какие промпты получил."""

    def __init__(self, reply="ответ модели"):
        self.reply, self.prompts = reply, []

    def __call__(self, msgs, max_new_tokens):
        self.prompts.append(msgs[-1]["content"])
        return [{"generated_text": msgs + [{"role": "assistant", "content": self.reply}]}]


class FakeEmbedder:
    """Вектор = сколько раз в тексте встречается каждое слово из WORDS."""

    def __init__(self):
        self.calls = 0

    def _vec(self, text):
        v = [text.lower().count(w) for w in WORDS]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def encode(self, texts, normalize_embeddings=True):
        self.calls += 1
        if isinstance(texts, str):
            return self._vec(texts)
        return [self._vec(t) for t in texts]
