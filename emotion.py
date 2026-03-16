import io
from typing import Optional, Tuple

import streamlit as st
from transformers import pipeline


SUPPORTED_FILE_TYPES = (".txt", ".md")


@st.cache_resource
def load_sentiment_pipeline():
    """Load and cache the HuggingFace sentiment analysis pipeline."""
    return pipeline("sentiment-analysis")


def extract_text_from_upload(uploaded_file) -> str:
    """Read supported uploaded text files and return their contents."""
    if uploaded_file is None:
        return ""

    file_name = uploaded_file.name.lower()
    if not file_name.endswith(SUPPORTED_FILE_TYPES):
        raise ValueError("Неподдерживаемый тип файла. Загрузите файл в формате .txt или .md.")

    file_bytes = uploaded_file.read()
    text_stream = io.BytesIO(file_bytes)
    return text_stream.read().decode("utf-8").strip()


def resolve_text_source(text_input: str, uploaded_file) -> Tuple[str, str]:
    """Pick file content over text input when both are available."""
    if uploaded_file is not None:
        extracted_text = extract_text_from_upload(uploaded_file)
        return extracted_text, "uploaded file"

    return text_input.strip(), "text input"


def analyze_sentiment(text: str) -> Optional[dict]:
    """Run sentiment analysis for the provided text."""
    if not text:
        return None

    sentiment_pipeline = load_sentiment_pipeline()
    results = sentiment_pipeline(text)
    return results[0] if results else None


def render_result(result: Optional[dict], source_name: str) -> None:
    """Display the sentiment analysis output."""
    if not result:
        st.warning(f"В {source_name} не найден текст. Добавьте текст для анализа.")
        return

    sentiment = result["label"]
    confidence = result["score"]

    st.subheader("Результат анализа")
    st.write(f"Источник: {source_name}")
    st.write(f"Тональность: {sentiment}")
    st.write(f"Уровень уверенности: {confidence:.4f}")


def main() -> None:
    st.set_page_config(page_title="Анализ тональности", layout="centered")
    st.title("Анализ тональности")
    st.write("Вставьте текст или загрузите файл `.txt` / `.md`, чтобы определить его тональность.")

    text_input = st.text_area("Введите текст", height=220, placeholder="Вставьте текст сюда...")
    uploaded_file = st.file_uploader("Загрузите файл", type=["txt", "md"])

    if st.button("Анализировать текст"):
        try:
            text_to_analyze, source_name = resolve_text_source(text_input, uploaded_file)
            result = analyze_sentiment(text_to_analyze)
            render_result(result, source_name)
        except UnicodeDecodeError:
            st.error("Не удалось декодировать загруженный файл как UTF-8 текст.")
        except ValueError as error:
            st.error(str(error))
        except Exception as error:
            st.error(f"Произошла непредвиденная ошибка: {error}")


if __name__ == "__main__":
    main()
