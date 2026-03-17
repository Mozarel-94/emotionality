import html
import io
import re
from typing import Dict, List, Optional, Sequence, Tuple

import streamlit as st
from transformers import pipeline


SUPPORTED_FILE_TYPES = (".txt", ".md")
MAX_SENTIMENT_CHARS = 1200
MAX_CLASSIFICATION_CHARS = 1800
MAX_FRAGMENT_CHARS = 400
MAX_FRAGMENTS_TO_SCORE = 8

EMOTION_LABELS = [
    "радость",
    "грусть",
    "злость",
    "страх",
    "удивление",
    "любовь",
    "интерес",
    "спокойствие",
]

TOPIC_LABELS = [
    "технологии",
    "бизнес",
    "финансы",
    "образование",
    "здоровье",
    "политика",
    "развлечения",
    "спорт",
    "путешествия",
    "наука",
    "отношения",
    "покупки",
]

SENTIMENT_TRANSLATIONS = {
    "POSITIVE": "Позитивная",
    "NEGATIVE": "Негативная",
    "NEUTRAL": "Нейтральная",
    "positive": "Позитивная",
    "negative": "Негативная",
    "neutral": "Нейтральная",
}


@st.cache_resource
def load_sentiment_pipeline():
    """Загружает и кеширует пайплайн анализа тональности."""
    return pipeline(
        "sentiment-analysis",
        model="blanchefort/rubert-base-cased-sentiment",
    )


@st.cache_resource
def load_zero_shot_pipeline():
    """Загружает и кеширует мультиязычный zero-shot классификатор."""
    return pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    )


def extract_text_from_upload(uploaded_file) -> str:
    """Считывает содержимое поддерживаемого текстового файла."""
    if uploaded_file is None:
        return ""

    file_name = uploaded_file.name.lower()
    if not file_name.endswith(SUPPORTED_FILE_TYPES):
        raise ValueError("Неподдерживаемый тип файла. Загрузите файл в формате .txt или .md.")

    file_bytes = uploaded_file.read()
    return io.BytesIO(file_bytes).read().decode("utf-8").strip()


def collect_sources(text_input: str, uploaded_files) -> List[Tuple[str, str]]:
    """Собирает список источников для анализа."""
    if uploaded_files:
        return [(uploaded_file.name, extract_text_from_upload(uploaded_file)) for uploaded_file in uploaded_files]

    return [("Текст из поля ввода", text_input.strip())]


def split_text_into_chunks(text: str, max_chars: int) -> List[str]:
    """Делит длинный текст на короткие части, чтобы не перегружать модель."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    chunks: List[str] = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > max_chars:
            sentence = sentence[:max_chars]

        candidate = f"{current_chunk} {sentence}".strip()
        if current_chunk and len(candidate) > max_chars:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk = candidate

    if current_chunk:
        chunks.append(current_chunk)

    return chunks or [normalized[:max_chars]]


def translate_sentiment_label(label: str) -> str:
    """Переводит метку тональности на русский."""
    return SENTIMENT_TRANSLATIONS.get(label.upper(), label)


def analyze_sentiment(text: str) -> Optional[Dict]:
    """Выполняет устойчивый анализ тональности даже для длинного текста."""
    chunks = split_text_into_chunks(text, MAX_SENTIMENT_CHARS)
    if not chunks:
        return None

    sentiment_pipeline = load_sentiment_pipeline()
    results = sentiment_pipeline(chunks)
    if not results:
        return None

    aggregated_scores: Dict[str, List[float]] = {}
    for result in results:
        label = result["label"]
        aggregated_scores.setdefault(label, []).append(float(result["score"]))

    best_label = max(
        aggregated_scores,
        key=lambda label: sum(aggregated_scores[label]) / len(aggregated_scores[label]),
    )
    average_score = sum(aggregated_scores[best_label]) / len(aggregated_scores[best_label])
    return {"label": translate_sentiment_label(best_label), "score": average_score}


def classify_labels(text: str, candidate_labels: Sequence[str], limit: int = 3) -> List[Dict]:
    """Определяет наиболее вероятные эмоции или темы."""
    chunks = split_text_into_chunks(text, MAX_CLASSIFICATION_CHARS)
    if not chunks:
        return []

    zero_shot_pipeline = load_zero_shot_pipeline()
    score_totals = {label: 0.0 for label in candidate_labels}
    score_counts = {label: 0 for label in candidate_labels}

    for chunk in chunks[:2]:
        result = zero_shot_pipeline(chunk, list(candidate_labels), multi_label=True)
        for label, score in zip(result["labels"], result["scores"]):
            score_totals[label] += float(score)
            score_counts[label] += 1

    ranked = []
    for label in candidate_labels:
        count = score_counts[label]
        if count:
            ranked.append({"label": label, "score": score_totals[label] / count})

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:limit]


def split_into_fragments(text: str) -> List[str]:
    """Разбивает текст на фрагменты для подсветки."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    fragments = re.split(r"(?<=[.!?])\s+", normalized)
    return [fragment.strip() for fragment in fragments if fragment.strip()]


def score_fragment(fragment: str, reference_labels: Sequence[str]) -> float:
    """Оценивает значимость фрагмента через zero-shot или упрощенный fallback."""
    if not fragment:
        return 0.0

    try:
        if reference_labels:
            zero_shot_pipeline = load_zero_shot_pipeline()
            result = zero_shot_pipeline(
                fragment[:MAX_FRAGMENT_CHARS],
                list(reference_labels),
                multi_label=True,
            )
            if result["scores"]:
                return float(max(result["scores"]))
    except Exception:
        pass

    return min(len(fragment) / 200, 1.0)


def select_important_fragments(text: str, topics: List[Dict], emotions: List[Dict]) -> List[Dict]:
    """Выбирает наиболее значимые фрагменты текста."""
    fragments = split_into_fragments(text)
    if not fragments:
        return []

    reference_labels = [item["label"] for item in topics[:2]] + [item["label"] for item in emotions[:2]]
    ranked_fragments = []

    for fragment in fragments[:MAX_FRAGMENTS_TO_SCORE]:
        ranked_fragments.append(
            {
                "text": fragment,
                "score": score_fragment(fragment, reference_labels),
            }
        )

    if not ranked_fragments:
        return []

    ranked_fragments.sort(key=lambda item: item["score"], reverse=True)
    return ranked_fragments[:3]


def build_highlighted_text(text: str, fragments: List[Dict]) -> str:
    """Формирует HTML с подсвеченными важными фрагментами."""
    highlighted_text = html.escape(text)

    for fragment in fragments:
        fragment_text = fragment["text"].strip()
        if not fragment_text:
            continue

        escaped_fragment = html.escape(fragment_text)
        highlighted_fragment = (
            "<mark style='background-color:#ffe08a; padding:0.1rem 0.2rem;'>"
            f"{escaped_fragment}"
            "</mark>"
        )
        highlighted_text = highlighted_text.replace(escaped_fragment, highlighted_fragment, 1)

    formatted_text = highlighted_text.replace("\n", "<br>")
    return f"<div class='review-highlight'>{formatted_text}</div>"


def format_top_labels(items: List[Dict]) -> str:
    """Преобразует метки и оценки в строку для интерфейса."""
    if not items:
        return "Не удалось определить"

    return ", ".join(f"{item['label']} ({item['score']:.2f})" for item in items)


def render_analysis_card(source_name: str, text: str) -> None:
    """Отображает результаты анализа для одного источника."""
    if not text:
        st.warning(f"В источнике `{source_name}` не найден текст. Добавьте содержимое для анализа.")
        return

    with st.container(border=True):
        st.subheader(source_name)

        sentiment = None
        try:
            sentiment = analyze_sentiment(text)
        except Exception as error:
            st.error(f"Не удалось выполнить анализ тональности: {error}")

        if sentiment:
            st.write(f"Тональность: {sentiment['label']}")
            st.write(f"Уровень уверенности: {sentiment['score']:.4f}")
        else:
            st.warning("Тональность не определена. Попробуйте сократить текст или проверить установку моделей.")

        emotions: List[Dict] = []
        topics: List[Dict] = []

        try:
            emotions = classify_labels(text, EMOTION_LABELS)
        except Exception:
            st.info("Эмоции временно недоступны, но базовый анализ тональности выполнен.")

        try:
            topics = classify_labels(text, TOPIC_LABELS)
        except Exception:
            st.info("Темы временно недоступны, но базовый анализ тональности выполнен.")

        st.write(f"Эмоции: {format_top_labels(emotions)}")
        st.write(f"Темы: {format_top_labels(topics)}")

        important_fragments = select_important_fragments(text, topics, emotions)
        if important_fragments:
            st.markdown("**Важные фрагменты**")
            st.markdown(build_highlighted_text(text, important_fragments), unsafe_allow_html=True)
            for index, fragment in enumerate(important_fragments, start=1):
                st.caption(f"Фрагмент {index}: значимость {fragment['score']:.2f}")


def main() -> None:
    st.set_page_config(page_title="Анализ тональности и эмоций", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 900px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stTextArea, .stFileUploader, .stButton {
            max-width: 760px;
            margin-left: auto;
            margin-right: auto;
        }
        h1, p, .stMarkdown, .stAlert, .stCaption, .stSubheader {
            max-width: 760px;
            margin-left: auto;
            margin-right: auto;
        }
        div[data-testid="stFileUploader"] > section,
        div[data-testid="stTextArea"] > div,
        div[data-testid="stButton"] > button {
            width: 100%;
        }
        div[data-testid="stButton"] > button {
            display: block;
            max-width: 320px;
            margin-left: auto;
            margin-right: auto;
        }
        .review-highlight {
            max-width: 760px;
            margin-left: auto;
            margin-right: auto;
            padding: 0.9rem 1rem;
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 0.5rem;
            background: rgba(250, 250, 250, 0.9);
            color: inherit;
            font-family: inherit;
            font-size: 1rem;
            line-height: 1.6;
            white-space: normal;
            word-break: break-word;
        }
        .review-highlight mark {
            color: inherit;
            font-family: inherit;
            font-size: inherit;
            line-height: inherit;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Анализ текста: тональность, эмоции и темы")
    st.write(
        "Вставьте текст или загрузите один или несколько файлов `.txt` / `.md`. "
        "Если файлы загружены, будет проанализировано их содержимое."
    )

    text_input = st.text_area("Введите текст", height=240, placeholder="Вставьте текст сюда...")
    uploaded_files = st.file_uploader(
        "Загрузите один или несколько файлов",
        type=["txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("Анализировать текст"):
        try:
            sources = collect_sources(text_input, uploaded_files)
            if not sources or all(not text for _, text in sources):
                st.warning("Не найден текст для анализа. Введите текст или загрузите файл.")
                return

            with st.spinner("Выполняем анализ текста..."):
                for source_name, text in sources:
                    render_analysis_card(source_name, text)
        except UnicodeDecodeError:
            st.error("Не удалось декодировать загруженный файл как UTF-8 текст.")
        except ValueError as error:
            st.error(str(error))
        except Exception as error:
            st.error(f"Произошла непредвиденная ошибка: {error}")


if __name__ == "__main__":
    main()
