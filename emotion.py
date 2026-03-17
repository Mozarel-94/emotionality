import html
import io
import csv
import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import altair as alt
import streamlit as st
from transformers import pipeline


SUPPORTED_FILE_TYPES = (".txt", ".md", ".csv", ".json")
MAX_SENTIMENT_CHARS = 1200
MAX_CLASSIFICATION_CHARS = 1800
MAX_FRAGMENT_CHARS = 400
MAX_FRAGMENTS_TO_SCORE = 8
MAX_REPEATED_PHRASES = 3
REVIEW_SEPARATOR = "\n---\n"

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


Document = Dict[str, Any]
AnalysisResult = Dict[str, Any]
REVIEW_TEXT_KEYS = ("review", "text", "comment", "content", "message", "feedback", "body")

RUSSIAN_STOPWORDS = {
    "а", "без", "более", "был", "была", "были", "было", "быть", "в", "вам", "вас", "весь", "во",
    "вот", "все", "всего", "вы", "где", "да", "даже", "для", "до", "его", "ее", "если", "есть",
    "еще", "же", "за", "здесь", "и", "из", "или", "им", "их", "к", "как", "ко", "когда", "кто",
    "ли", "либо", "мне", "можно", "мой", "мы", "на", "над", "нам", "нас", "не", "него", "нее",
    "нет", "ни", "них", "но", "ну", "о", "об", "однако", "он", "она", "они", "оно", "от", "очень",
    "по", "под", "при", "с", "со", "так", "также", "такой", "там", "те", "тем", "то", "того",
    "тоже", "только", "том", "ты", "у", "уже", "хотя", "чего", "чей", "чем", "что", "чтобы", "эта",
    "эти", "это", "я",
}

APP_PALETTE = {
    "bg": "#f3f7fb",
    "surface": "rgba(255, 255, 255, 0.86)",
    "surface_strong": "#ffffff",
    "border": "rgba(15, 23, 42, 0.08)",
    "text": "#102033",
    "muted": "#62748a",
    "accent": "#2f6fed",
    "accent_soft": "#e4edff",
    "positive": "#22a06b",
    "positive_soft": "#dff6ea",
    "neutral": "#f0b44c",
    "neutral_soft": "#fff1d6",
    "negative": "#d95c5c",
    "negative_soft": "#fde4e4",
    "shadow": "0 22px 60px rgba(15, 23, 42, 0.09)",
}

SENTIMENT_UI = {
    "positive": {
        "label": "Позитивный фон",
        "color": APP_PALETTE["positive"],
        "soft": APP_PALETTE["positive_soft"],
    },
    "neutral": {
        "label": "Нейтральный фон",
        "color": APP_PALETTE["neutral"],
        "soft": APP_PALETTE["neutral_soft"],
    },
    "negative": {
        "label": "Негативный фон",
        "color": APP_PALETTE["negative"],
        "soft": APP_PALETTE["negative_soft"],
    },
}


@st.cache_resource
def load_sentiment_pipeline():
    """Р—Р°РіСЂСѓР¶Р°РµС‚ Рё РєРµС€РёСЂСѓРµС‚ РїР°Р№РїР»Р°Р№РЅ Р°РЅР°Р»РёР·Р° С‚РѕРЅР°Р»СЊРЅРѕСЃС‚Рё."""
    return pipeline(
        "sentiment-analysis",
        model="blanchefort/rubert-base-cased-sentiment",
    )


@st.cache_resource
def load_zero_shot_pipeline():
    """Р—Р°РіСЂСѓР¶Р°РµС‚ Рё РєРµС€РёСЂСѓРµС‚ РјСѓР»СЊС‚РёСЏР·С‹С‡РЅС‹Р№ zero-shot РєР»Р°СЃСЃРёС„РёРєР°С‚РѕСЂ."""
    return pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    )


def decode_uploaded_file(uploaded_file) -> str:
    """Считывает содержимое файла как UTF-8 текст."""
    file_bytes = uploaded_file.read()
    return io.BytesIO(file_bytes).read().decode("utf-8").strip()


def flatten_json_to_text(data: Any) -> str:
    """Преобразует JSON-структуру в плоский текст для анализа."""
    if data is None:
        return ""
    if isinstance(data, str):
        return data.strip()
    if isinstance(data, (int, float, bool)):
        return str(data)
    if isinstance(data, list):
        parts = [flatten_json_to_text(item) for item in data]
        return "\n".join(part for part in parts if part)
    if isinstance(data, dict):
        parts = []
        for key, value in data.items():
            value_text = flatten_json_to_text(value)
            if value_text:
                parts.append(f"{key}: {value_text}")
        return "\n".join(parts)
    return str(data).strip()


def csv_text_to_text(csv_text: str) -> str:
    """Преобразует CSV в читаемый текстовый вид для анализа."""
    reader = csv.reader(io.StringIO(csv_text))
    rows = []
    for row in reader:
        normalized_row = " | ".join(cell.strip() for cell in row if cell and cell.strip())
        if normalized_row:
            rows.append(normalized_row)
    return "\n".join(rows)


def split_reviews_from_text(text: str) -> List[str]:
    """Разбивает текст на отдельные отзывы."""
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []

    # Рекомендуемый формат: один отзыв на блок, разделитель ---
    if "\n---\n" in normalized:
        parts = [part.strip() for part in normalized.split(REVIEW_SEPARATOR)]
        return [part for part in parts if part]

    # Fallback: пустая строка между отзывами.
    parts = [part.strip() for part in re.split(r"\n\s*\n+", normalized) if part.strip()]
    if len(parts) > 1:
        return parts

    return [normalized]


def extract_review_text_from_record(record: Dict[str, Any]) -> str:
    """Извлекает текст отзыва из словаря по типовым полям."""
    lowered = {str(key).lower(): value for key, value in record.items()}
    for key in REVIEW_TEXT_KEYS:
        if key in lowered and lowered[key] is not None:
            return str(lowered[key]).strip()
    return flatten_json_to_text(record)


def build_review_documents(
    reviews: List[str],
    source_name: str,
    source_type: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """Строит список документов, по одному на отзыв."""
    documents: List[Document] = []
    base_metadata = metadata or {}
    for index, review_text in enumerate(reviews, start=1):
        if not review_text.strip():
            continue
        document_name = source_name if len(reviews) == 1 else f"{source_name} · отзыв {index}"
        document_metadata = {**base_metadata, "review_index": index, "reviews_total": len(reviews)}
        documents.append(build_document(document_name, source_type, review_text, document_metadata))
    return documents


def parse_csv_reviews(csv_text: str, source_name: str) -> List[Document]:
    """Извлекает отдельные отзывы из CSV."""
    reader = csv.DictReader(io.StringIO(csv_text))
    documents: List[Document] = []

    if reader.fieldnames:
        for index, row in enumerate(reader, start=1):
            review_text = extract_review_text_from_record(row)
            if review_text:
                documents.append(
                    build_document(
                        f"{source_name} · отзыв {index}",
                        "csv",
                        review_text,
                        metadata={"format": "csv", "review_index": index},
                    )
                )

    if documents:
        return documents

    fallback_reviews = split_reviews_from_text(csv_text_to_text(csv_text))
    return build_review_documents(fallback_reviews, source_name, "csv", metadata={"format": "csv"})


def parse_json_reviews(json_data: Any, source_name: str) -> List[Document]:
    """Извлекает отдельные отзывы из JSON."""
    records: List[Any]

    if isinstance(json_data, dict) and isinstance(json_data.get("reviews"), list):
        records = json_data["reviews"]
    elif isinstance(json_data, list):
        records = json_data
    else:
        records = [json_data]

    documents: List[Document] = []
    for index, record in enumerate(records, start=1):
        if isinstance(record, dict):
            review_text = extract_review_text_from_record(record)
            metadata = {"format": "json", "review_index": index, "raw_data": record}
        else:
            review_text = flatten_json_to_text(record)
            metadata = {"format": "json", "review_index": index, "raw_data": record}

        if review_text:
            documents.append(
                build_document(
                    f"{source_name} · отзыв {index}",
                    "json",
                    review_text,
                    metadata=metadata,
                )
            )

    return documents


def build_document(name: str, source_type: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> Document:
    """Создает единый внутренний формат документа для анализа."""
    return {
        "name": name,
        "source_type": source_type,
        "text": text.strip(),
        "metadata": metadata or {},
    }


def extract_documents_from_upload(uploaded_file) -> List[Document]:
    """Нормализует загруженный файл в список документов, по одному на отзыв."""
    if uploaded_file is None:
        return [build_document("Пустой источник", "unknown", "")]

    file_name = uploaded_file.name
    lower_name = file_name.lower()
    if not lower_name.endswith(SUPPORTED_FILE_TYPES):
        raise ValueError("Неподдерживаемый тип файла. Загрузите файл в формате .txt, .md, .csv или .json.")

    raw_text = decode_uploaded_file(uploaded_file)
    if lower_name.endswith(".json"):
        parsed_json = json.loads(raw_text) if raw_text else {}
        documents = parse_json_reviews(parsed_json, file_name)
        return documents or build_review_documents(
            [flatten_json_to_text(parsed_json)],
            file_name,
            "json",
            metadata={"format": "json", "raw_data": parsed_json},
        )

    if lower_name.endswith(".csv"):
        return parse_csv_reviews(raw_text, file_name)

    extension = lower_name.rsplit(".", 1)[-1]
    return build_review_documents(
        split_reviews_from_text(raw_text),
        file_name,
        extension,
        metadata={"format": extension},
    )


def collect_sources(text_input: str, uploaded_files) -> List[Document]:
    """Собирает список документов для анализа в едином формате."""
    if uploaded_files:
        documents: List[Document] = []
        for uploaded_file in uploaded_files:
            documents.extend(extract_documents_from_upload(uploaded_file))
        return documents

    return build_review_documents(
        split_reviews_from_text(text_input),
        "Текст из поля ввода",
        "text_input",
        metadata={"format": "text"},
    )

def split_text_into_chunks(text: str, max_chars: int) -> List[str]:
    """Р”РµР»РёС‚ РґР»РёРЅРЅС‹Р№ С‚РµРєСЃС‚ РЅР° РєРѕСЂРѕС‚РєРёРµ С‡Р°СЃС‚Рё, С‡С‚РѕР±С‹ РЅРµ РїРµСЂРµРіСЂСѓР¶Р°С‚СЊ РјРѕРґРµР»СЊ."""
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
    """РџРµСЂРµРІРѕРґРёС‚ РјРµС‚РєСѓ С‚РѕРЅР°Р»СЊРЅРѕСЃС‚Рё РЅР° СЂСѓСЃСЃРєРёР№."""
    return SENTIMENT_TRANSLATIONS.get(label.upper(), label)


def analyze_sentiment(text: str) -> Optional[Dict]:
    """Р’С‹РїРѕР»РЅСЏРµС‚ СѓСЃС‚РѕР№С‡РёРІС‹Р№ Р°РЅР°Р»РёР· С‚РѕРЅР°Р»СЊРЅРѕСЃС‚Рё РґР°Р¶Рµ РґР»СЏ РґР»РёРЅРЅРѕРіРѕ С‚РµРєСЃС‚Р°."""
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
    """РћРїСЂРµРґРµР»СЏРµС‚ РЅР°РёР±РѕР»РµРµ РІРµСЂРѕСЏС‚РЅС‹Рµ СЌРјРѕС†РёРё РёР»Рё С‚РµРјС‹."""
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
    """Р Р°Р·Р±РёРІР°РµС‚ С‚РµРєСЃС‚ РЅР° С„СЂР°РіРјРµРЅС‚С‹ РґР»СЏ РїРѕРґСЃРІРµС‚РєРё."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    fragments = re.split(r"(?<=[.!?])\s+", normalized)
    return [fragment.strip() for fragment in fragments if fragment.strip()]


def score_fragment(fragment: str, reference_labels: Sequence[str]) -> float:
    """РћС†РµРЅРёРІР°РµС‚ Р·РЅР°С‡РёРјРѕСЃС‚СЊ С„СЂР°РіРјРµРЅС‚Р° С‡РµСЂРµР· zero-shot РёР»Рё СѓРїСЂРѕС‰РµРЅРЅС‹Р№ fallback."""
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
    """Р’С‹Р±РёСЂР°РµС‚ РЅР°РёР±РѕР»РµРµ Р·РЅР°С‡РёРјС‹Рµ С„СЂР°РіРјРµРЅС‚С‹ С‚РµРєСЃС‚Р°."""
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
    """Р¤РѕСЂРјРёСЂСѓРµС‚ HTML СЃ РїРѕРґСЃРІРµС‡РµРЅРЅС‹РјРё РІР°Р¶РЅС‹РјРё С„СЂР°РіРјРµРЅС‚Р°РјРё."""
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
    """РџСЂРµРѕР±СЂР°Р·СѓРµС‚ РјРµС‚РєРё Рё РѕС†РµРЅРєРё РІ СЃС‚СЂРѕРєСѓ РґР»СЏ РёРЅС‚РµСЂС„РµР№СЃР°."""
    if not items:
        return "Не удалось определить"

    return ", ".join(f"{item['label']} ({item['score']:.2f})" for item in items)


def get_sentiment_ui(bucket: str) -> Dict[str, str]:
    """Возвращает цветовую схему для типа тональности."""
    return SENTIMENT_UI.get(bucket, SENTIMENT_UI["neutral"])


def render_badge(label: str, tone: str = "neutral") -> str:
    """Создает HTML-бейдж для компактной визуальной маркировки."""
    palette = get_sentiment_ui(tone)
    return (
        f"<span class='ui-badge' style='background:{palette['soft']};color:{palette['color']};"
        f"border-color:{palette['color']}22'>{html.escape(label)}</span>"
    )


def render_metric_card(title: str, value: str, caption: str, tone: str = "neutral") -> str:
    """Создает HTML-карточку для KPI."""
    palette = get_sentiment_ui(tone)
    return f"""
    <div class="metric-card fade-in">
        <div class="metric-card__eyebrow">
            <span class="metric-card__dot" style="background:{palette['color']};"></span>
            {html.escape(title)}
        </div>
        <div class="metric-card__value">{html.escape(value)}</div>
        <div class="metric-card__caption">{html.escape(caption)}</div>
    </div>
    """


def render_insight_card(title: str, body: str, tone: str = "neutral") -> str:
    """Создает карточку с кратким выводом."""
    palette = get_sentiment_ui(tone)
    return f"""
    <div class="insight-card fade-in" style="background:{palette['soft']}; border-color:{palette['color']}22;">
        <div class="insight-card__title" style="color:{palette['color']};">{html.escape(title)}</div>
        <div class="insight-card__body">{html.escape(body)}</div>
    </div>
    """


def build_sentiment_distribution_chart(counts: Dict[str, int]):
    """Строит диаграмму распределения отзывов по тональности."""
    chart_data = [
        {"tone": "Негативные", "count": counts["negative"], "color": APP_PALETTE["negative"]},
        {"tone": "Нейтральные", "count": counts["neutral"], "color": APP_PALETTE["neutral"]},
        {"tone": "Позитивные", "count": counts["positive"], "color": APP_PALETTE["positive"]},
    ]
    return (
        alt.Chart(alt.Data(values=chart_data))
        .mark_bar(cornerRadiusTopLeft=10, cornerRadiusTopRight=10, size=52)
        .encode(
            x=alt.X("tone:N", title=None, sort=None, axis=alt.Axis(labelAngle=0, labelColor=APP_PALETTE["muted"])),
            y=alt.Y("count:Q", title="Количество отзывов", axis=alt.Axis(gridColor="#e5edf7")),
            color=alt.Color("color:N", scale=None, legend=None),
            tooltip=[
                alt.Tooltip("tone:N", title="Тональность"),
                alt.Tooltip("count:Q", title="Количество"),
            ],
        )
        .properties(height=260)
        .configure_view(strokeOpacity=0)
        .configure(background="transparent")
    )


def build_topic_chart(analyses: List[AnalysisResult]):
    """Строит диаграмму по самым частым темам."""
    topic_counter: Counter[str] = Counter()
    for analysis in analyses:
        for topic in analysis.get("topics", [])[:2]:
            topic_counter[topic["label"]] += float(topic["score"])

    if not topic_counter:
        return None

    chart_data = [
        {"topic": topic, "score": round(score, 2)}
        for topic, score in topic_counter.most_common(6)
    ]
    return (
        alt.Chart(alt.Data(values=chart_data))
        .mark_bar(cornerRadius=10, color=APP_PALETTE["accent"])
        .encode(
            y=alt.Y("topic:N", title=None, sort="-x", axis=alt.Axis(labelColor=APP_PALETTE["muted"])),
            x=alt.X("score:Q", title="Суммарный вес темы", axis=alt.Axis(gridColor="#e5edf7")),
            tooltip=[
                alt.Tooltip("topic:N", title="Тема"),
                alt.Tooltip("score:Q", title="Вес"),
            ],
        )
        .properties(height=260)
        .configure_view(strokeOpacity=0)
        .configure(background="transparent")
    )


def build_emotion_chart(analyses: List[AnalysisResult]):
    """Строит диаграмму по самым заметным эмоциям."""
    emotion_counter: Counter[str] = Counter()
    for analysis in analyses:
        for emotion in analysis.get("emotions", [])[:2]:
            emotion_counter[emotion["label"]] += float(emotion["score"])

    if not emotion_counter:
        return None

    chart_data = [
        {"emotion": emotion, "score": round(score, 2)}
        for emotion, score in emotion_counter.most_common(6)
    ]
    return (
        alt.Chart(alt.Data(values=chart_data))
        .mark_arc(innerRadius=56, outerRadius=96)
        .encode(
            theta=alt.Theta("score:Q"),
            color=alt.Color(
                "emotion:N",
                scale=alt.Scale(
                    range=[
                        APP_PALETTE["accent"],
                        APP_PALETTE["positive"],
                        APP_PALETTE["neutral"],
                        "#7c67ff",
                        "#1aa7a1",
                        "#ff8f5a",
                    ]
                ),
                legend=alt.Legend(title="Эмоции", orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("emotion:N", title="Эмоция"),
                alt.Tooltip("score:Q", title="Вес"),
            ],
        )
        .properties(height=260)
        .configure_view(strokeOpacity=0)
        .configure(background="transparent")
    )


def sentiment_to_bucket(sentiment: Optional[Dict]) -> str:
    """Нормализует тональность в одну из трех корзин."""
    if not sentiment:
        return "neutral"

    label = str(sentiment.get("label", "")).lower()
    if "позитив" in label:
        return "positive"
    if "негатив" in label:
        return "negative"
    return "neutral"


def sentiment_to_value(sentiment: Optional[Dict]) -> float:
    """Преобразует тональность в шкалу от -1 до 1."""
    if not sentiment:
        return 0.0

    score = float(sentiment.get("score", 0.0))
    bucket = sentiment_to_bucket(sentiment)
    if bucket == "positive":
        return score
    if bucket == "negative":
        return -score
    return 0.0


def collect_repeated_phrases(fragments: List[str]) -> List[str]:
    """Ищет повторяющиеся короткие фразы в наборе фрагментов."""
    phrase_counter: Counter[str] = Counter()

    for fragment in fragments:
        tokens = re.findall(r"[а-яА-Яa-zA-ZёЁ]{3,}", fragment.lower())
        tokens = [token for token in tokens if token not in RUSSIAN_STOPWORDS]
        for size in (2, 3):
            for index in range(len(tokens) - size + 1):
                phrase = " ".join(tokens[index:index + size])
                phrase_counter[phrase] += 1

    return [
        phrase
        for phrase, count in phrase_counter.most_common()
        if count > 1
    ][:MAX_REPEATED_PHRASES]


def build_repeated_insights(analyses: List[AnalysisResult]) -> List[str]:
    """Формирует короткие выводы о повторяющихся замечаниях."""
    positive_fragments: List[str] = []
    negative_fragments: List[str] = []
    positive_topics: Counter[str] = Counter()
    negative_topics: Counter[str] = Counter()

    for analysis in analyses:
        bucket = sentiment_to_bucket(analysis.get("sentiment"))
        fragments = [item["text"] for item in analysis.get("important_fragments", []) if item.get("text")]
        topics = analysis.get("topics", [])

        if bucket == "positive":
            positive_fragments.extend(fragments)
            if topics:
                positive_topics[topics[0]["label"]] += 1
        elif bucket == "negative":
            negative_fragments.extend(fragments)
            if topics:
                negative_topics[topics[0]["label"]] += 1

    insights: List[str] = []
    positive_phrases = collect_repeated_phrases(positive_fragments)
    negative_phrases = collect_repeated_phrases(negative_fragments)

    if positive_phrases:
        insights.append(f"Посетители часто хвалят: {positive_phrases[0]}.")
    elif positive_topics:
        insights.append(f"Среди положительных отзывов чаще всего встречается тема: {positive_topics.most_common(1)[0][0]}.")

    if negative_phrases:
        insights.append(f"Посетители чаще всего жалуются на: {negative_phrases[0]}.")
    elif negative_topics:
        insights.append(f"Среди негативных отзывов чаще всего встречается тема: {negative_topics.most_common(1)[0][0]}.")

    if not insights:
        all_fragments = [
            item["text"]
            for analysis in analyses
            for item in analysis.get("important_fragments", [])
            if item.get("text")
        ]
        common_phrases = collect_repeated_phrases(all_fragments)
        if common_phrases:
            insights.append(f"Чаще всего в отзывах повторяется: {common_phrases[0]}.")

    return insights[:3]


def get_custom_css() -> str:
    """Возвращает кастомные стили приложения."""
    return f"""
    <style>
    :root {{
        --bg: {APP_PALETTE["bg"]};
        --surface: {APP_PALETTE["surface"]};
        --surface-strong: {APP_PALETTE["surface_strong"]};
        --border: {APP_PALETTE["border"]};
        --text: {APP_PALETTE["text"]};
        --muted: {APP_PALETTE["muted"]};
        --accent: {APP_PALETTE["accent"]};
        --accent-soft: {APP_PALETTE["accent_soft"]};
        --positive: {APP_PALETTE["positive"]};
        --neutral: {APP_PALETTE["neutral"]};
        --negative: {APP_PALETTE["negative"]};
        --shadow: {APP_PALETTE["shadow"]};
    }}

    .stApp {{
        background:
            radial-gradient(circle at top left, rgba(47, 111, 237, 0.14), transparent 28%),
            radial-gradient(circle at top right, rgba(34, 160, 107, 0.12), transparent 24%),
            linear-gradient(180deg, #f8fbff 0%, var(--bg) 48%, #eef4fb 100%);
        color: var(--text);
    }}

    .block-container {{
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }}

    html, body, [class*="css"] {{
        font-family: "Aptos", "Segoe UI", "Trebuchet MS", sans-serif;
    }}

    h1, h2, h3 {{
        color: var(--text);
        letter-spacing: -0.02em;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: var(--surface);
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        border-radius: 24px;
        backdrop-filter: blur(16px);
        transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
        transform: translateY(-2px);
        box-shadow: 0 24px 66px rgba(15, 23, 42, 0.12);
        border-color: rgba(47, 111, 237, 0.16);
    }}

    div[data-testid="stTextArea"] textarea {{
        min-height: 260px;
        border-radius: 20px;
        border: 1px solid rgba(47, 111, 237, 0.16);
        background: rgba(255,255,255,0.82);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.85);
        font-size: 1rem;
        line-height: 1.65;
        color: var(--text);
    }}

    div[data-testid="stTextArea"] textarea:focus {{
        border-color: rgba(47, 111, 237, 0.45);
        box-shadow: 0 0 0 4px rgba(47, 111, 237, 0.10);
    }}

    div[data-testid="stFileUploader"] > section {{
        border-radius: 20px;
        border: 1px dashed rgba(47, 111, 237, 0.24);
        background: rgba(255,255,255,0.72);
    }}

    div[data-testid="stFileUploader"] section:hover {{
        border-color: rgba(47, 111, 237, 0.45);
        background: rgba(255,255,255,0.92);
    }}

    div[data-testid="stButton"] > button {{
        width: 100%;
        min-height: 3.4rem;
        border-radius: 18px;
        border: none;
        background: linear-gradient(135deg, var(--accent) 0%, #4a8bff 100%);
        color: white;
        font-weight: 700;
        letter-spacing: 0.01em;
        box-shadow: 0 16px 28px rgba(47, 111, 237, 0.22);
        transition: transform 180ms ease, box-shadow 180ms ease, filter 180ms ease;
    }}

    div[data-testid="stButton"] > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 18px 32px rgba(47, 111, 237, 0.28);
        filter: brightness(1.03);
    }}

    div[data-testid="stTabs"] button {{
        border-radius: 999px;
        padding: 0.5rem 1rem;
        color: var(--muted);
    }}

    div[data-testid="stTabs"] button[aria-selected="true"] {{
        background: var(--surface-strong);
        color: var(--text);
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
    }}

    .app-hero {{
        display: block;
        margin-bottom: 1.4rem;
    }}

    .hero-panel, .hero-aside, .metric-card, .insight-card, .micro-card {{
        background: var(--surface);
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        border-radius: 24px;
        backdrop-filter: blur(16px);
    }}

    .hero-panel {{
        padding: 1.6rem 1.7rem;
        max-width: 620px;
    }}

    .hero-aside {{
        padding: 1.3rem 1.4rem;
    }}

    .eyebrow {{
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.35rem 0.65rem;
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--accent);
        font-size: 0.82rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }}

    .eyebrow::before {{
        content: "";
        width: 0.5rem;
        height: 0.5rem;
        border-radius: 50%;
        background: var(--accent);
        box-shadow: 0 0 0 4px rgba(47,111,237,0.12);
    }}

    .hero-title {{
        font-size: clamp(2.3rem, 4vw, 4rem);
        line-height: 0.98;
        margin: 0;
        max-width: 12ch;
    }}

    .app-name {{
        margin: 0;
        font-size: clamp(3rem, 6vw, 5.2rem);
        line-height: 0.92;
        letter-spacing: -0.05em;
        font-weight: 900;
        color: var(--text);
        text-wrap: balance;
    }}

    .hero-text, .aside-text, .section-text, .metric-card__caption, .insight-card__body {{
        color: var(--muted);
        line-height: 1.7;
    }}

    .chip-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        margin-top: 0.9rem;
    }}

    .chip {{
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.5rem 0.8rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.88);
        border: 1px solid var(--border);
        color: var(--text);
        font-size: 0.9rem;
        font-weight: 600;
    }}

    .chip::before {{
        content: "";
        width: 0.45rem;
        height: 0.45rem;
        border-radius: 50%;
        background: var(--accent);
    }}

    .mini-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.8rem;
        margin-top: 1rem;
    }}

    .micro-card {{
        padding: 0.85rem 0.95rem;
    }}

    .micro-card__label {{
        color: var(--muted);
        font-size: 0.84rem;
        margin-bottom: 0.35rem;
    }}

    .micro-card__value {{
        font-size: 1.35rem;
        font-weight: 800;
        color: var(--text);
    }}

    .section-title {{
        font-size: 1.35rem;
        margin-bottom: 0.15rem;
    }}

    .ui-badge {{
        display: inline-flex;
        align-items: center;
        padding: 0.3rem 0.65rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
        border: 1px solid transparent;
    }}

    .metric-card {{
        padding: 1.05rem 1.1rem;
        min-height: 148px;
    }}

    .metric-card__eyebrow {{
        display: flex;
        align-items: center;
        gap: 0.45rem;
        color: var(--muted);
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}

    .metric-card__dot {{
        width: 0.6rem;
        height: 0.6rem;
        border-radius: 50%;
    }}

    .metric-card__value {{
        margin-top: 0.7rem;
        font-size: 2rem;
        font-weight: 800;
        color: var(--text);
        letter-spacing: -0.03em;
    }}

    .insight-card {{
        padding: 1rem 1.1rem;
        min-height: 124px;
    }}

    .insight-card__title {{
        font-size: 0.88rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.55rem;
    }}

    .chart-shell {{
        background: rgba(255,255,255,0.76);
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 0.8rem 0.9rem 0.2rem 0.9rem;
    }}

    .review-highlight {{
        margin-top: 0.85rem;
        padding: 1rem 1.05rem;
        border-radius: 18px;
        border: 1px solid rgba(15,23,42,0.08);
        background: rgba(248, 251, 255, 0.92);
        line-height: 1.7;
    }}

    .review-highlight mark {{
        color: inherit;
        border-radius: 0.5rem;
        box-shadow: inset 0 -0.75rem 0 rgba(240, 180, 76, 0.28);
    }}

    .section-anchor {{
        margin-top: 1.6rem;
    }}

    .sidebar-stack {{
        display: flex;
        flex-direction: column;
        gap: 0.8rem;
    }}

    .fade-in {{
        animation: fadeUp 420ms ease both;
    }}

    @keyframes fadeUp {{
        from {{
            opacity: 0;
            transform: translateY(12px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    @media (max-width: 980px) {{
        .app-hero {{
            grid-template-columns: 1fr;
        }}
    }}
    </style>
    """


def render_app_hero() -> None:
    """Отображает верхний hero-блок приложения."""
    st.markdown(
        """
        <div class="app-hero fade-in">
            <div class="hero-panel">
                <div class="eyebrow">Приложение</div>
                <h1 class="app-name">Маркер эмоций</h1>
                <div class="chip-row">
                    <span class="chip">Тональность</span>
                    <span class="chip">Темы и эмоции</span>
                    <span class="chip">Сводные инсайты</span>
                    <span class="chip">Поддержка нескольких файлов</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_input_sidebar(text_input: str, uploaded_files) -> None:
    """Показывает сопроводительную панель рядом с вводом."""
    files_count = len(uploaded_files) if uploaded_files else 0
    char_count = len(text_input.strip())
    text_state = "Готов к анализу" if text_input.strip() or files_count else "Ожидает данные"

    st.markdown(
        f"""
        <div class="hero-aside fade-in">
            <div class="section-title">Статус</div>
            <div class="mini-grid">
                <div class="micro-card">
                    <div class="micro-card__label">Символов</div>
                    <div class="micro-card__value">{char_count}</div>
                </div>
                <div class="micro-card">
                    <div class="micro-card__label">Файлов</div>
                    <div class="micro-card__value">{files_count}</div>
                </div>
            </div>
            <div class="chip-row">
                <span class="ui-badge" style="background:{APP_PALETTE['accent_soft']};color:{APP_PALETTE['accent']};border-color:{APP_PALETTE['accent']}22;">{text_state}</span>
                <span class="ui-badge" style="background:#f8fbff;color:{APP_PALETTE['muted']};border-color:{APP_PALETTE['border']};">Форматы: TXT, MD, CSV, JSON</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def analyze_document(document: Document) -> AnalysisResult:
    """Выполняет полный анализ одного документа и возвращает результат."""
    text = str(document.get("text", "")).strip()

    sentiment = None
    emotions: List[Dict] = []
    topics: List[Dict] = []
    important_fragments: List[Dict] = []
    errors: List[str] = []

    if text:
        try:
            sentiment = analyze_sentiment(text)
        except Exception as error:
            errors.append(f"Не удалось выполнить анализ тональности: {error}")

        try:
            emotions = classify_labels(text, EMOTION_LABELS)
        except Exception:
            errors.append("Эмоции временно недоступны.")

        try:
            topics = classify_labels(text, TOPIC_LABELS)
        except Exception:
            errors.append("Темы временно недоступны.")

        try:
            important_fragments = select_important_fragments(text, topics, emotions)
        except Exception:
            errors.append("Не удалось выделить важные фрагменты.")

    return {
        "document": document,
        "sentiment": sentiment,
        "emotions": emotions,
        "topics": topics,
        "important_fragments": important_fragments,
        "errors": errors,
    }


def render_summary_dashboard(analyses: List[AnalysisResult]) -> None:
    """Отображает сводную аналитику по всем отзывам."""
    if not analyses:
        return

    counts = {"positive": 0, "neutral": 0, "negative": 0}
    total_value = 0.0

    for analysis in analyses:
        bucket = sentiment_to_bucket(analysis.get("sentiment"))
        counts[bucket] += 1
        total_value += sentiment_to_value(analysis.get("sentiment"))

    average_value = total_value / len(analyses)
    marker_position = max(0.0, min(100.0, (average_value + 1) * 50))
    sentiment_label = "Смешанный фон"
    if average_value > 0.2:
        sentiment_label = "Преимущественно позитивные отзывы"
    elif average_value < -0.2:
        sentiment_label = "Преимущественно негативные отзывы"

    insights = build_repeated_insights(analyses)
    topic_chart = build_topic_chart(analyses)
    emotion_chart = build_emotion_chart(analyses)
    sentiment_chart = build_sentiment_distribution_chart(counts)
    total_reviews = len(analyses)
    avg_confidence = 0.0
    scored_sentiments = [analysis["sentiment"]["score"] for analysis in analyses if analysis.get("sentiment")]
    if scored_sentiments:
        avg_confidence = sum(scored_sentiments) / len(scored_sentiments)

    with st.container(border=True):
        st.markdown("<div class='section-title'>Сводный анализ отзывов</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="chart-shell fade-in" style="margin-top:1rem;margin-bottom:1rem;">
                <div style="font-size:0.95rem;margin-bottom:0.45rem;"><strong>Общая эмоциональность</strong>: {sentiment_label}</div>
                <div style="position:relative;height:18px;border-radius:999px;background:linear-gradient(90deg,{APP_PALETTE['negative']} 0%,{APP_PALETTE['neutral']} 50%,{APP_PALETTE['positive']} 100%);overflow:hidden;">
                    <div style="position:absolute;left:calc({marker_position}% - 9px);top:-3px;width:18px;height:24px;border-radius:999px;background:#1f2937;border:3px solid white;box-shadow:0 2px 8px rgba(0,0,0,0.2);"></div>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:0.85rem;margin-top:0.35rem;color:#4b5563;">
                    <span>Негативные</span>
                    <span>Нейтральные</span>
                    <span>Позитивные</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        kpi_cols = st.columns(4)
        with kpi_cols[0]:
            st.markdown(render_metric_card("Всего отзывов", str(total_reviews), "Обработано источников", "neutral"), unsafe_allow_html=True)
        with kpi_cols[1]:
            st.markdown(render_metric_card("Позитивные", str(counts["positive"]), "С положительным фоном", "positive"), unsafe_allow_html=True)
        with kpi_cols[2]:
            st.markdown(render_metric_card("Нейтральные", str(counts["neutral"]), "С ровной эмоциональностью", "neutral"), unsafe_allow_html=True)
        with kpi_cols[3]:
            st.markdown(render_metric_card("Средняя уверенность", f"{avg_confidence:.2f}", "Средний confidence моделей", "negative" if average_value < 0 else "positive"), unsafe_allow_html=True)

        chart_col1, chart_col2 = st.columns(2, gap="large")
        with chart_col1:
            with st.container(border=True):
                st.markdown("**Распределение тональности**")
                st.altair_chart(sentiment_chart, use_container_width=True)
        with chart_col2:
            with st.container(border=True):
                st.markdown("**Самые заметные темы**")
                if topic_chart is not None:
                    st.altair_chart(topic_chart, use_container_width=True)
                else:
                    st.info("Недостаточно данных, чтобы построить диаграмму тем.")

        lower_col1, lower_col2 = st.columns([1.15, 0.85], gap="large")
        with lower_col1:
            with st.container(border=True):
                st.markdown("**Эмоциональный профиль**")
                if emotion_chart is not None:
                    st.altair_chart(emotion_chart, use_container_width=True)
                else:
                    st.info("Недостаточно данных, чтобы построить диаграмму эмоций.")
        with lower_col2:
            st.markdown("**Повторяющиеся замечания**")
            if insights:
                insight_tones = ["positive", "negative", "neutral"]
                for index, insight in enumerate(insights):
                    st.markdown(
                        render_insight_card(
                            f"Инсайт {index + 1}",
                            insight,
                            insight_tones[min(index, len(insight_tones) - 1)],
                        ),
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    render_insight_card(
                        "Инсайты пока не сформированы",
                        "Недостаточно повторяющихся отзывов, чтобы выделить устойчивые замечания.",
                        "neutral",
                    ),
                    unsafe_allow_html=True,
                )


def render_analysis_card(analysis: AnalysisResult) -> None:
    """Отображает результаты анализа для одного документа."""
    document = analysis["document"]
    source_name = str(document.get("name", "Источник"))
    text = str(document.get("text", "")).strip()
    metadata = document.get("metadata", {}) or {}
    source_type = str(document.get("source_type", "unknown"))

    if not text:
        st.warning(f"В источнике `{source_name}` не найден текст. Добавьте содержимое для анализа.")
        return

    sentiment = analysis.get("sentiment")
    bucket = sentiment_to_bucket(sentiment)
    sentiment_ui = get_sentiment_ui(bucket)
    top_emotion = analysis.get("emotions", [{}])[0].get("label", "Нет данных") if analysis.get("emotions") else "Нет данных"
    top_topic = analysis.get("topics", [{}])[0].get("label", "Нет данных") if analysis.get("topics") else "Нет данных"
    sentiment_score = float(sentiment.get("score", 0.0)) if sentiment else 0.0

    with st.container(border=True):
        st.markdown(
            f"""
            <div class="fade-in" style="display:flex;justify-content:space-between;gap:1rem;align-items:flex-start;margin-bottom:0.8rem;">
                <div>
                    <div class="section-title" style="margin-bottom:0.2rem;">{html.escape(source_name)}</div>
                    <div class="section-text">Формат: {html.escape(str(metadata.get('format', source_type)))}</div>
                </div>
                <div>{render_badge(sentiment_ui['label'], bucket)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        metric_cols = st.columns(3)
        with metric_cols[0]:
            st.markdown(
                render_metric_card(
                    "Тональность",
                    sentiment["label"] if sentiment else "Нет данных",
                    "Итоговое настроение текста",
                    bucket,
                ),
                unsafe_allow_html=True,
            )
        with metric_cols[1]:
            st.markdown(
                render_metric_card(
                    "Главная эмоция",
                    top_emotion,
                    "Лидирующий эмоциональный сигнал",
                    bucket,
                ),
                unsafe_allow_html=True,
            )
        with metric_cols[2]:
            st.markdown(
                render_metric_card(
                    "Главная тема",
                    top_topic,
                    "Чаще всего заметная тема",
                    bucket,
                ),
                unsafe_allow_html=True,
            )

        if sentiment:
            st.markdown("**Индикатор уверенности**")
            st.progress(min(max(sentiment_score, 0.0), 1.0), text=f"{sentiment['score']:.2%}")
        else:
            st.warning("Тональность не определена. Попробуйте сократить текст или проверить установку моделей.")

        emotions = analysis.get("emotions", [])
        topics = analysis.get("topics", [])

        st.write(f"Эмоции: {format_top_labels(emotions)}")
        st.write(f"Темы: {format_top_labels(topics)}")

        important_fragments = analysis.get("important_fragments", [])
        if important_fragments:
            st.markdown("**Важные фрагменты**")
            st.markdown(build_highlighted_text(text, important_fragments), unsafe_allow_html=True)
            for index, fragment in enumerate(important_fragments, start=1):
                st.caption(f"Фрагмент {index}: значимость {fragment['score']:.2f}")

        for error_message in analysis.get("errors", []):
            st.info(error_message)


def main() -> None:
    st.set_page_config(page_title="Анализ тональности и эмоций", layout="wide")
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    render_app_hero()

    input_col, support_col = st.columns([1.22, 0.78], gap="medium")
    with input_col:
        with st.container(border=True):
            st.markdown("<div class='section-title'>Ввод данных</div>", unsafe_allow_html=True)
            text_input = st.text_area(
                "Введите текст",
                height=260,
                placeholder="Один отзыв на блок. Для нескольких отзывов используйте разделитель ---",
                label_visibility="visible",
                help="TXT/MD: один отзыв на блок, разделитель --- или пустая строка.",
            )
            uploaded_files = st.file_uploader(
                "Загрузите один или несколько файлов",
                type=["txt", "md", "csv", "json"],
                accept_multiple_files=True,
                help="CSV: один отзыв в строке. JSON: массив reviews или список объектов с полем review/text/comment.",
            )
            analyze_trigger = st.button("Запустить анализ")

    with support_col:
        st.markdown("<div class='sidebar-stack'>", unsafe_allow_html=True)
        render_input_sidebar(text_input, uploaded_files)
        with st.container(border=True):
            st.markdown("<div class='section-title'>Шаги</div>", unsafe_allow_html=True)
            st.markdown(
                render_insight_card(
                    "1. Добавьте данные",
                    "2. Запустите анализ  3. Откройте сводку или отзывы",
                    "positive",
                ),
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    if analyze_trigger:
        try:
            sources = collect_sources(text_input, uploaded_files)
            if not sources or all(not str(document.get("text", "")).strip() for document in sources):
                st.warning("Не найден текст для анализа. Введите текст или загрузите файл.")
                return

            with st.spinner("Анализ..."):
                analyses: List[AnalysisResult] = []
                progress = st.progress(0.0, text="Обработка...")
                for index, document in enumerate(sources, start=1):
                    analyses.append(analyze_document(document))
                    progress.progress(index / len(sources), text=f"Обработано отзывов: {index} из {len(sources)}")
                progress.empty()

                st.markdown("<div class='section-anchor'></div>", unsafe_allow_html=True)
                result_tabs = st.tabs(["Сводка", "Отзывы"])
                with result_tabs[0]:
                    render_summary_dashboard(analyses)
                with result_tabs[1]:
                    st.markdown(
                        "<div class='section-title'>Отзывы</div>",
                        unsafe_allow_html=True,
                    )
                    for analysis in analyses:
                        render_analysis_card(analysis)
        except UnicodeDecodeError:
            st.error("Не удалось декодировать загруженный файл как UTF-8 текст.")
        except ValueError as error:
            st.error(str(error))
        except Exception as error:
            st.error(f"Произошла непредвиденная ошибка: {error}")


if __name__ == "__main__":
    main()


