# 🎓 Ассистент по лекциям

Приложение превращает аудиозапись лекции в текст, делает по нему конспект и отвечает на вопросы студента **только по материалу лекции**.

Проектный практикум, КГЭУ, кафедра «Информационные технологии и интеллектуальные системы».
Выполнил: Карманов Н. В., группа ИИНм-1-26. Руководитель: к.т.н., доцент Хамитов Р. М.

## Как это работает

```
аудио лекции ──► Whisper ──► текст лекции ──┬──► Qwen ──► конспект
                                            │
вопрос студента ──► E5: поиск 3 ближайших ──┴──► Qwen ──► ответ по фрагментам
                    кусков лекции
```

| Задача | Модель | Размер |
|---|---|---|
| Речь → текст | [openai/whisper-small](https://huggingface.co/openai/whisper-small) | 244 млн параметров |
| Конспект и ответы | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | 1,5 млрд параметров |
| Поиск по смыслу | [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | 118 млн параметров |

Ответ на вопрос строится по схеме **RAG** (Retrieval-Augmented Generation): лекция режется на куски по 500 символов, E5 находит 3 куска, ближайших к вопросу по смыслу, и Qwen отвечает только по ним. Если ответа во фрагментах нет, модель должна сказать «В лекции об этом не говорится».

## Структура

```
lecture-assistant/
├── assistant.py          # логика: три модели, нарезка текста, поиск, конспект
├── api.py                # FastAPI-сервер: /transcribe, /summary, /ask, /health
├── app.py                # Streamlit-интерфейс, обращается к серверу по HTTP
├── tests/                # pytest: логика и API на заглушках моделей
├── notebooks/
│   ├── lecture_assistant_demo.ipynb   # демонстрация моделей в Colab
│   └── run_in_colab.ipynb             # запуск приложения в Colab
└── requirements.txt
```

## Демонстрация в Google Colab

Ноутбук с пошаговой работой трёх моделей: [открыть в Colab](https://colab.research.google.com/drive/19wORuTzF8zzzOcX_ykMCVtZpCm-GDVmJ)

## Запуск приложения

Моделям нужна видеокарта, поэтому удобнее всего запускать в Colab (бесплатный GPU T4):
[открыть `notebooks/run_in_colab.ipynb` в Colab](https://colab.research.google.com/github/ageraforreal/lecture-assistant/blob/main/notebooks/run_in_colab.ipynb), выбрать GPU T4 и запустить все ячейки. Последняя выдаст публичную ссылку на приложение.

Локально (нужны Python 3.10–3.12, ffmpeg, желательно видеокарта NVIDIA):

```bash
pip install -r requirements.txt
uvicorn api:app --port 8000          # терминал 1: сервер
streamlit run app.py                 # терминал 2: интерфейс, http://localhost:8501
```

Если сервер запущен на другом адресе: `API_URL=http://адрес:8000 streamlit run app.py`.

## API

Документация Swagger доступна на `http://localhost:8000/docs`.

| Метод | Путь | Вход | Выход |
|---|---|---|---|
| GET | `/health` | - | `{"status": "ok"}` |
| POST | `/transcribe` | аудиофайл (mp3, wav, m4a, ogg, flac) | `{"text", "chars"}` |
| POST | `/summary` | `{"text"}` | `{"summary"}` |
| POST | `/ask` | `{"text", "question"}` | `{"answer", "fragments": [{"text", "score"}]}` |

Пример:

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"text": "текст лекции...", "question": "Что такое гиппокамп?"}'
```

## Тесты

```bash
python -m pytest -v
```

19 тестов проверяют нарезку текста, конспект длинной лекции, поиск фрагментов, кэширование векторов и все эндпоинты API, включая ошибки (пустой текст, не аудиофайл). Нейросети в тестах заменены заглушками из `tests/fakes.py`, поэтому тесты идут меньше секунды и не требуют GPU.

## Ограничения

- Whisper-small ошибается в терминах («клиентивные» вместо «когнитивные»). Модель `whisper-medium` точнее, но в 3 раза медленнее.
- Qwen 1.5B иногда добавляет факты не из лекции. Против этого в промпте есть запрет отвечать вне фрагментов.
- Без видеокарты распознавание и генерация работают в разы медленнее, поэтому основной способ запуска - Colab с GPU.
