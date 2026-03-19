import html
import io
import csv
import json
import os
import re
from collections import Counter
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request
from xml.etree import ElementTree

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
DEFAULT_HTTP_TIMEOUT = 12
MAX_SEARCH_RESULTS = 10
MAX_SEARCH_QUERIES = 6
MAX_PAGE_TEXT_CHARS = 6000
SEARCH_RESULT_KEYS = ("organic", "organic_results", "results", "items")
BLOCKED_SCHEMES = ("mailto:", "javascript:", "tel:")
SEARCH_ENGINE_DOMAINS = (
    "duckduckgo.com",
    "www.duckduckgo.com",
    "google.com",
    "www.google.com",
    "ya.ru",
    "yandex.ru",
    "www.yandex.ru",
    "bing.com",
    "www.bing.com",
)
ANTI_BOT_MARKERS = (
    "Unfortunately, bots use DuckDuckGo too",
    "anomaly-modal",
    "Please complete the following challenge",
)
NEWSAPI_MAX_RESULTS = 100

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


class HTMLTextExtractor(HTMLParser):
    """Извлекает текст и базовые мета-теги из HTML."""

    def __init__(self) -> None:
        super().__init__()
        self.in_script = False
        self.in_style = False
        self.in_title = False
        self.text_parts: List[str] = []
        self.title_parts: List[str] = []
        self.meta: Dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        normalized_tag = tag.lower()
        attrs_dict = {str(key).lower(): value for key, value in attrs}
        if normalized_tag == "script":
            self.in_script = True
        elif normalized_tag == "style":
            self.in_style = True
        elif normalized_tag == "title":
            self.in_title = True
        elif normalized_tag == "meta":
            meta_key = str(attrs_dict.get("property") or attrs_dict.get("name") or "").lower()
            meta_content = str(attrs_dict.get("content") or "").strip()
            if meta_key and meta_content:
                self.meta[meta_key] = meta_content

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.lower()
        if normalized_tag == "script":
            self.in_script = False
        elif normalized_tag == "style":
            self.in_style = False
        elif normalized_tag == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        if self.in_script or self.in_style:
            return

        cleaned = re.sub(r"\s+", " ", data).strip()
        if not cleaned:
            return

        if self.in_title:
            self.title_parts.append(cleaned)
        self.text_parts.append(cleaned)


class DuckDuckGoResultsParser(HTMLParser):
    """Извлекает ссылки и сниппеты из HTML-выдачи DuckDuckGo."""

    def __init__(self) -> None:
        super().__init__()
        self.results: List[Dict[str, str]] = []
        self.current_result: Optional[Dict[str, str]] = None
        self.capture_title = False
        self.capture_snippet = False
        self.capture_source = False

    def _flush_current_result(self) -> None:
        if self.current_result and self.current_result.get("url"):
            self.results.append(self.current_result)
        self.current_result = None
        self.capture_title = False
        self.capture_snippet = False
        self.capture_source = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attrs_dict = {str(key).lower(): str(value or "") for key, value in attrs}
        css_class = attrs_dict.get("class", "")

        if tag.lower() == "a" and "result__a" in css_class:
            href = attrs_dict.get("href", "").strip()
            if href:
                self._flush_current_result()
                self.current_result = {"title": "", "url": href, "snippet": "", "source": ""}
                self.capture_title = True
        elif tag.lower() in ("a", "div", "span") and ("result__snippet" in css_class or "result__extras__url" in css_class):
            if self.current_result:
                if "result__snippet" in css_class:
                    self.capture_snippet = True
                if "result__extras__url" in css_class:
                    self.capture_source = True

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.lower()
        if normalized_tag == "a" and self.capture_title:
            self.capture_title = False
        elif normalized_tag in ("a", "div", "span"):
            self.capture_snippet = False
            self.capture_source = False

    def handle_data(self, data: str) -> None:
        cleaned = re.sub(r"\s+", " ", data).strip()
        if not cleaned or not self.current_result:
            return

        if self.capture_title:
            self.current_result["title"] = f"{self.current_result['title']} {cleaned}".strip()
        elif self.capture_snippet:
            self.current_result["snippet"] = f"{self.current_result['snippet']} {cleaned}".strip()
        elif self.capture_source:
            self.current_result["source"] = f"{self.current_result['source']} {cleaned}".strip()


def split_comma_lines(raw_value: str) -> List[str]:
    """Разбивает ввод по запятым и строкам."""
    if not raw_value.strip():
        return []
    values = [item.strip() for item in re.split(r"[\n,;]+", raw_value) if item.strip()]
    return list(dict.fromkeys(values))


def normalize_match_text(text: str) -> str:
    """Готовит текст к поиску совпадений."""
    lowered = text.lower().replace("ё", "е")
    lowered = re.sub(r"[\"'`«»“”„]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def build_search_queries(entity_name: str, aliases: List[str], extra_terms: List[str]) -> List[str]:
    """Формирует набор поисковых запросов для поиска упоминаний."""
    candidates = [entity_name.strip()] + [alias.strip() for alias in aliases if alias.strip()]
    candidates = list(dict.fromkeys([candidate for candidate in candidates if candidate]))
    query_suffix = " ".join(extra_terms[:3]).strip()
    queries: List[str] = []

    for candidate in candidates[:MAX_SEARCH_QUERIES]:
        quoted = f"\"{candidate}\""
        queries.append(f"{quoted} {query_suffix}".strip())
        if query_suffix:
            queries.append(candidate)

    return list(dict.fromkeys(queries))[:MAX_SEARCH_QUERIES]


def build_newsapi_query(entity_name: str, aliases: List[str]) -> str:
    """Строит запрос для NewsAPI по названию юрлица и алиасам."""
    main_name = entity_name.strip()
    normalized_aliases = [alias.strip() for alias in aliases if alias.strip() and alias.strip().lower() != main_name.lower()]
    if not main_name:
        return ""

    main_expr = f'"{main_name}"'
    if not normalized_aliases:
        return main_expr

    alias_expr = " OR ".join(f'"{alias}"' for alias in normalized_aliases)
    return f"{main_expr} OR ({alias_expr})"


def extract_domain(url: str) -> str:
    """Возвращает домен URL без www."""
    try:
        host = urllib_parse.urlparse(url).netloc.lower().strip()
    except Exception:
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host


def is_allowed_url(url: str, allowed_domains: List[str], blocked_domains: List[str]) -> bool:
    """Проверяет, можно ли использовать URL для парсинга."""
    normalized_url = url.strip().lower()
    if not normalized_url or normalized_url.startswith(BLOCKED_SCHEMES):
        return False

    parsed = urllib_parse.urlparse(normalized_url)
    if parsed.scheme not in ("http", "https"):
        return False

    domain = extract_domain(normalized_url)
    if not domain:
        return False
    if domain in SEARCH_ENGINE_DOMAINS:
        return False

    normalized_allowed = [item.lower() for item in allowed_domains if item.strip()]
    normalized_blocked = [item.lower() for item in blocked_domains if item.strip()]

    if any(domain == blocked or domain.endswith(f".{blocked}") for blocked in normalized_blocked):
        return False
    if normalized_allowed and not any(domain == allowed or domain.endswith(f".{allowed}") for allowed in normalized_allowed):
        return False
    return True


def load_search_config() -> Dict[str, str]:
    """Читает настройки поискового провайдера из переменных окружения."""
    provider = os.getenv("SEARCH_PROVIDER", "auto").strip().lower() or "auto"
    return {
        "provider": provider,
        "api_key": os.getenv("SEARCH_API_KEY", "").strip(),
        "base_url": os.getenv("SEARCH_BASE_URL", "").strip(),
    }


def load_newsapi_config() -> Dict[str, str]:
    """Читает настройки NewsAPI из переменных окружения."""
    return {
        "api_key": os.getenv("NEWSAPI_API_KEY", "435e8dcc837e438a9692120d1373b285").strip(),
        "base_url": os.getenv("NEWSAPI_BASE_URL", "https://newsapi.org/v2/everything").strip() or "https://newsapi.org/v2/everything",
    }


def build_request(url: str, method: str = "GET", data: Optional[bytes] = None, headers: Optional[Dict[str, str]] = None):
    """Создает HTTP-запрос с безопасным user-agent."""
    request_headers = {
        "User-Agent": "Mozilla/5.0 (compatible; EmotionalityBot/1.0; +https://localhost)",
        "Accept-Language": "ru,en;q=0.9",
    }
    if headers:
        request_headers.update(headers)
    return urllib_request.Request(url, data=data, headers=request_headers, method=method)


def fetch_json(url: str, method: str = "GET", payload: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Выполняет HTTP-запрос и возвращает JSON."""
    body = None
    request_headers = dict(headers or {})
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")

    request = build_request(url, method=method, data=body, headers=request_headers)
    with urllib_request.urlopen(request, timeout=DEFAULT_HTTP_TIMEOUT) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        raw_text = response.read().decode(charset, errors="replace")
    parsed = json.loads(raw_text)
    return parsed if isinstance(parsed, dict) else {"items": parsed}


def fetch_url_text(url: str) -> str:
    """Скачивает HTML как текст."""
    request = build_request(url)
    with urllib_request.urlopen(request, timeout=DEFAULT_HTTP_TIMEOUT) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def is_anti_bot_page(text: str) -> bool:
    """Определяет, что вместо выдачи пришла anti-bot страница."""
    return any(marker in text for marker in ANTI_BOT_MARKERS)


def parse_search_results(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    """Нормализует ответ поискового провайдера."""
    raw_results: List[Any] = []
    for key in SEARCH_RESULT_KEYS:
        if isinstance(payload.get(key), list):
            raw_results = payload[key]
            break
    if not raw_results and isinstance(payload.get("data"), list):
        raw_results = payload["data"]

    results: List[Dict[str, str]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        url = str(item.get("link") or item.get("url") or "").strip()
        if not url:
            continue
        title = str(item.get("title") or item.get("name") or "").strip()
        snippet = str(item.get("snippet") or item.get("description") or item.get("text") or "").strip()
        source_name = str(item.get("source") or item.get("displayed_link") or extract_domain(url)).strip()
        results.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
                "source": source_name,
            }
        )
    return results


def parse_bing_rss_results(xml_text: str) -> List[Dict[str, str]]:
    """Парсит RSS-выдачу Bing в единый формат результатов."""
    results: List[Dict[str, str]] = []
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return results

    channel = root.find("channel")
    if channel is None:
        return results

    for item in channel.findall("item"):
        title = (item.findtext("title") or "").strip()
        url = (item.findtext("link") or "").strip()
        snippet = (item.findtext("description") or "").strip()
        if not url or not is_allowed_url(url, [], []):
            continue
        results.append(
            {
                "title": html.unescape(title),
                "url": url,
                "snippet": html.unescape(snippet),
                "source": extract_domain(url),
            }
        )
    return results


def normalize_bing_result_url(url: str) -> str:
    """Преобразует redirect-ссылку Bing в прямую внешнюю ссылку."""
    cleaned = html.unescape(url.strip())
    if not cleaned:
        return ""

    parsed = urllib_parse.urlparse(cleaned)
    if "bing.com" not in parsed.netloc:
        return cleaned

    params = urllib_parse.parse_qs(parsed.query)
    encoded_values = params.get("u", [])
    if not encoded_values:
        return cleaned

    encoded = encoded_values[0]
    if encoded.startswith("a1"):
        encoded = encoded[2:]

    padding = "=" * (-len(encoded) % 4)
    try:
        import base64

        decoded = base64.urlsafe_b64decode(encoded + padding).decode("utf-8", errors="replace").strip()
        return decoded or cleaned
    except Exception:
        return cleaned


def parse_bing_html_results(html_text: str) -> List[Dict[str, str]]:
    """Парсит обычную HTML-выдачу Bing."""
    results: List[Dict[str, str]] = []
    pattern = re.compile(
        r'<li class="b_algo"[\s\S]*?<h2[^>]*><a[^>]+href="([^"]+)"[^>]*>([\s\S]*?)</a></h2>[\s\S]*?(?:<div class="b_caption"><p[^>]*>([\s\S]*?)</p>)?',
        re.IGNORECASE,
    )
    for raw_url, raw_title, raw_snippet in pattern.findall(html_text):
        url = normalize_bing_result_url(raw_url)
        title = re.sub(r"<[^>]+>", " ", raw_title)
        snippet = re.sub(r"<[^>]+>", " ", raw_snippet or "")
        title = re.sub(r"\s+", " ", html.unescape(title)).strip()
        snippet = re.sub(r"\s+", " ", html.unescape(snippet)).strip()
        if not url or not title:
            continue
        if not is_allowed_url(url, [], []):
            continue
        results.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
                "source": extract_domain(url),
            }
        )
    return results


def normalize_search_result_url(url: str) -> str:
    """Преобразует URL поисковой выдачи в прямую ссылку."""
    cleaned = html.unescape(url.strip())
    if not cleaned:
        return ""

    parsed = urllib_parse.urlparse(cleaned)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        params = urllib_parse.parse_qs(parsed.query)
        uddg = params.get("uddg", [])
        if uddg:
            return urllib_parse.unquote(uddg[0]).strip()
    return cleaned


def parse_duckduckgo_html_results(html_text: str) -> List[Dict[str, str]]:
    """Нормализует HTML-выдачу DuckDuckGo в список результатов поиска."""
    parser = DuckDuckGoResultsParser()
    parser.feed(html_text)
    parser.close()
    parser._flush_current_result()

    results: List[Dict[str, str]] = []
    for item in parser.results:
        url = normalize_search_result_url(item.get("url", ""))
        if not url:
            continue
        if not is_allowed_url(url, [], []):
            continue
        results.append(
            {
                "title": item.get("title", "").strip(),
                "url": url,
                "snippet": item.get("snippet", "").strip(),
                "source": item.get("source", "").strip() or extract_domain(url),
            }
        )
    if results:
        return results

    # Fallback для более простой HTML-выдачи, когда классы result__a/result__snippet отсутствуют.
    generic_results: List[Dict[str, str]] = []
    link_pattern = re.compile(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
    for href, raw_title in link_pattern.findall(html_text):
        url = normalize_search_result_url(re.sub(r"\s+", " ", href))
        title = re.sub(r"<[^>]+>", " ", raw_title)
        title = re.sub(r"\s+", " ", html.unescape(title)).strip()
        if not url or not title:
            continue
        if not is_allowed_url(url, [], []):
            continue
        generic_results.append(
            {
                "title": title,
                "url": url,
                "snippet": "",
                "source": extract_domain(url),
            }
        )
        if len(generic_results) >= MAX_SEARCH_RESULTS:
            break
    return generic_results


def search_web_bing_rss(query: str, max_results: int, base_url: str = "") -> List[Dict[str, str]]:
    """Ищет страницы через Bing RSS без API."""
    endpoint = base_url or "https://www.bing.com/search"
    params = urllib_parse.urlencode({"format": "rss", "q": query})
    xml_text = fetch_url_text(f"{endpoint}?{params}")
    return parse_bing_rss_results(xml_text)[:max_results]


def search_web_bing_html(query: str, max_results: int, base_url: str = "") -> List[Dict[str, str]]:
    """Ищет страницы через HTML-выдачу Bing без API."""
    endpoint = base_url or "https://www.bing.com/search"
    params = urllib_parse.urlencode({"q": query})
    html_text = fetch_url_text(f"{endpoint}?{params}")
    return parse_bing_html_results(html_text)[:max_results]


def search_web_duckduckgo(query: str, max_results: int, base_url: str = "") -> List[Dict[str, str]]:
    """Ищет страницы через HTML-выдачу DuckDuckGo."""
    endpoint = base_url or "https://html.duckduckgo.com/html/"
    params = urllib_parse.urlencode({"q": query, "kl": "ru-ru"})
    html_text = fetch_url_text(f"{endpoint}?{params}")
    if is_anti_bot_page(html_text):
        return []
    return parse_duckduckgo_html_results(html_text)[:max_results]


def search_web(query: str, max_results: int) -> List[Dict[str, str]]:
    """Ищет страницы через API или HTML-выдачу без API."""
    search_config = load_search_config()
    provider = search_config["provider"]
    api_key = search_config["api_key"]
    base_url = search_config["base_url"]

    if provider in ("", "auto"):
        provider = "auto"
    elif not api_key and provider in ("serpapi", "serper"):
        provider = "auto"

    if provider == "serpapi":
        endpoint = base_url or "https://serpapi.com/search.json"
        params = urllib_parse.urlencode({"engine": "google", "q": query, "api_key": api_key, "num": max_results, "hl": "ru"})
        payload = fetch_json(f"{endpoint}?{params}")
    elif provider == "serper":
        endpoint = base_url or "https://google.serper.dev/search"
        payload = fetch_json(
            endpoint,
            method="POST",
            payload={"q": query, "num": max_results, "gl": "ru", "hl": "ru"},
            headers={"X-API-KEY": api_key},
        )
    elif provider == "bing_rss":
        return search_web_bing_rss(query, max_results, base_url)
    elif provider == "bing_html":
        return search_web_bing_html(query, max_results, base_url)
    elif provider == "duckduckgo_html":
        return search_web_duckduckgo(query, max_results, base_url)
    elif provider == "auto":
        for fallback in (
            lambda: search_web_bing_html(query, max_results),
            lambda: search_web_bing_rss(query, max_results),
            lambda: search_web_duckduckgo(query, max_results),
        ):
            try:
                results = fallback()
            except Exception:
                results = []
            if results:
                return results
        return []
    else:
        if not base_url:
            raise ValueError("Для кастомного SEARCH_PROVIDER нужно задать SEARCH_BASE_URL.")
        params = urllib_parse.urlencode({"q": query, "api_key": api_key, "num": max_results})
        separator = "&" if "?" in base_url else "?"
        payload = fetch_json(f"{base_url}{separator}{params}")

    return parse_search_results(payload)


def parse_html_page(html_text: str) -> Dict[str, str]:
    """Извлекает заголовок, дату и текст из HTML."""
    extractor = HTMLTextExtractor()
    extractor.feed(html_text)
    extractor.close()

    title = " ".join(extractor.title_parts).strip()
    meta_title = extractor.meta.get("og:title") or extractor.meta.get("twitter:title") or extractor.meta.get("title") or ""
    if meta_title and len(meta_title) > len(title):
        title = meta_title

    description = extractor.meta.get("description") or extractor.meta.get("og:description") or extractor.meta.get("twitter:description") or ""
    published_at = (
        extractor.meta.get("article:published_time")
        or extractor.meta.get("og:published_time")
        or extractor.meta.get("publication_date")
        or extractor.meta.get("pubdate")
        or ""
    )

    raw_text = "\n".join(extractor.text_parts)
    cleaned_text = re.sub(r"\s+", " ", raw_text)
    cleaned_text = re.sub(r"(cookie|навигация|подписаться|войти|регистрация)(\s+\1)+", r"\1", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = cleaned_text.strip()[:MAX_PAGE_TEXT_CHARS]

    return {
        "title": title.strip(),
        "description": description.strip(),
        "published_at": published_at.strip(),
        "text": cleaned_text,
    }


def find_matched_terms(text: str, terms: Sequence[str]) -> List[str]:
    """Возвращает совпавшие термины из набора."""
    normalized_text = normalize_match_text(text)
    matched = []
    for term in terms:
        normalized_term = normalize_match_text(term)
        if normalized_term and normalized_term in normalized_text:
            matched.append(term.strip())
    return list(dict.fromkeys(matched))


def detect_supported_language(text: str) -> str:
    """Грубо определяет, относится ли текст к русскому или английскому языку."""
    sample = str(text or "").strip()[:2000]
    if not sample:
        return "unknown"

    cyrillic_count = len(re.findall(r"[А-Яа-яЁё]", sample))
    latin_count = len(re.findall(r"[A-Za-z]", sample))
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", sample))

    if cjk_count and cjk_count >= max(cyrillic_count, latin_count):
        return "other"
    if cyrillic_count >= max(8, latin_count):
        return "ru"
    if latin_count >= max(8, cyrillic_count):
        return "en"
    if cyrillic_count > 0 and latin_count > 0:
        return "mixed"
    return "unknown"


def evaluate_entity_match(
    entity_name: str,
    aliases: List[str],
    extra_terms: List[str],
    page_title: str,
    page_text: str,
    snippet: str,
    domain: str,
    allowed_domains: List[str],
) -> Tuple[str, List[str], str]:
    """Определяет релевантность страницы к указанному юрлицу."""
    base_terms = [entity_name] + aliases
    combined_text = " ".join([page_title, snippet, page_text, domain]).strip()
    language = detect_supported_language(" ".join([page_title, snippet, page_text]))
    if language not in ("ru", "en", "mixed"):
        return "rejected", [], language

    matched_terms = find_matched_terms(combined_text, base_terms)
    matched_extra = find_matched_terms(combined_text, extra_terms)

    score = 0
    if matched_terms:
        score += 2
    if entity_name.strip() and entity_name.strip() in matched_terms:
        score += 2
    if matched_extra:
        score += 1
    if allowed_domains and any(domain == allowed or domain.endswith(f".{allowed}") for allowed in allowed_domains):
        score += 1

    if not matched_terms:
        return "rejected", matched_extra, language
    if score >= 4:
        return "relevant", matched_terms + matched_extra, language
    return "possible", matched_terms + matched_extra, language


def build_internet_document(
    entity_name: str,
    search_query: str,
    search_result: Dict[str, str],
    page_payload: Dict[str, str],
    match_status: str,
    matched_terms: List[str],
    language: str = "unknown",
    fetch_error: str = "",
) -> Document:
    """Нормализует интернет-упоминание в текущий формат документа."""
    url = search_result.get("url", "")
    domain = extract_domain(url)
    page_title = page_payload.get("title", "").strip() or search_result.get("title", "").strip()
    page_text = page_payload.get("text", "").strip()
    text = page_text or " ".join(
        part for part in [page_title, page_payload.get("description", ""), search_result.get("snippet", "")]
        if part.strip()
    )
    return build_document(
        page_title or f"Упоминание: {entity_name}",
        "internet_mention",
        text,
        metadata={
            "format": "internet",
            "entity_name": entity_name,
            "url": url,
            "domain": domain,
            "search_query": search_query,
            "search_title": search_result.get("title", "").strip(),
            "search_snippet": search_result.get("snippet", "").strip(),
            "published_at": page_payload.get("published_at", "").strip(),
            "match_status": match_status,
            "matched_terms": matched_terms,
            "language": language,
            "fetch_error": fetch_error,
        },
    )


def build_news_document(
    entity_name: str,
    query: str,
    article: Dict[str, Any],
    match_status: str,
    matched_terms: List[str],
    language: str,
) -> Document:
    """Нормализует новостную статью в документ для анализа."""
    source_data = article.get("source", {}) if isinstance(article.get("source"), dict) else {}
    title = str(article.get("title", "") or "").strip()
    description = str(article.get("description", "") or "").strip()
    content = str(article.get("content", "") or "").strip()
    url = str(article.get("url", "") or "").strip()
    source_name = str(source_data.get("name", "") or "").strip() or extract_domain(url) or "Новостной источник"
    full_text = "\n".join(part for part in [title, description, content] if part)
    return build_document(
        title or f"Новость: {entity_name}",
        "news_mention",
        full_text,
        metadata={
            "format": "newsapi",
            "entity_name": entity_name,
            "url": url,
            "domain": extract_domain(url),
            "source_name": source_name,
            "author": str(article.get("author", "") or "").strip(),
            "published_at": str(article.get("publishedAt", "") or "").strip(),
            "language": language,
            "query": query,
            "match_status": match_status,
            "matched_terms": matched_terms,
            "search_title": title,
            "search_snippet": description,
        },
    )


def fetch_newsapi_articles(news_params: Dict[str, Any]) -> Dict[str, Any]:
    """Получает статьи из NewsAPI."""
    news_config = load_newsapi_config()
    api_key = news_config["api_key"]
    base_url = news_config["base_url"]
    if not api_key:
        raise ValueError("Не задан NEWSAPI_API_KEY. Добавьте ключ NewsAPI в переменные окружения.")

    query = build_newsapi_query(str(news_params.get("entity_name", "")), list(news_params.get("aliases", [])))
    if not query:
        raise ValueError("Укажите юридическое лицо для поиска новостей.")

    max_results = min(max(int(news_params.get("max_results", 10)), 1), NEWSAPI_MAX_RESULTS)
    query_params: Dict[str, Any] = {
        "q": query,
        "pageSize": max_results,
        "sortBy": "publishedAt",
    }
    language = str(news_params.get("language", "") or "").strip().lower()
    if language in ("ru", "en"):
        query_params["language"] = language
    date_from = str(news_params.get("date_from", "") or "").strip()
    date_to = str(news_params.get("date_to", "") or "").strip()
    if date_from:
        query_params["from"] = date_from
    if date_to:
        query_params["to"] = date_to

    params = urllib_parse.urlencode(query_params)
    payload = fetch_json(
        f"{base_url}?{params}",
        headers={"X-Api-Key": api_key},
    )
    if payload.get("status") == "error":
        raise ValueError(str(payload.get("message") or "NewsAPI вернул ошибку."))
    return payload


def collect_news_mentions(news_params: Dict[str, Any]) -> Tuple[List[Document], Dict[str, Any]]:
    """Ищет новости о юрлице через NewsAPI."""
    entity_name = str(news_params.get("entity_name", "")).strip()
    aliases = list(news_params.get("aliases", []))
    extra_terms = list(news_params.get("extra_terms", []))
    allowed_sources = [item.lower() for item in news_params.get("allowed_sources", []) if str(item).strip()]
    query = build_newsapi_query(entity_name, aliases)
    report: Dict[str, Any] = {
        "report_type": "news",
        "entity_name": entity_name,
        "queries": [query] if query else [],
        "search_errors": [],
        "fetch_errors": [],
        "rejected_pages": [],
        "possible_pages": [],
        "found_urls": 0,
        "parsed_urls": 0,
        "relevant_count": 0,
        "possible_count": 0,
        "rejected_count": 0,
        "source_stats": {},
    }

    if not entity_name:
        raise ValueError("Укажите юридическое лицо для поиска новостей.")

    payload = fetch_newsapi_articles(news_params)
    articles = payload.get("articles", [])
    if not isinstance(articles, list):
        articles = []
    report["found_urls"] = len(articles)

    documents: List[Document] = []
    source_counter: Counter[str] = Counter()
    for article in articles:
        if not isinstance(article, dict):
            continue
        source_data = article.get("source", {}) if isinstance(article.get("source"), dict) else {}
        source_name = str(source_data.get("name", "") or "").strip()
        source_id = str(source_data.get("id", "") or "").strip().lower()
        if allowed_sources and not any(candidate in allowed_sources for candidate in [source_id, source_name.lower()]):
            continue

        title = str(article.get("title", "") or "").strip()
        description = str(article.get("description", "") or "").strip()
        content = str(article.get("content", "") or "").strip()
        article_text = "\n".join(part for part in [title, description, content] if part)
        report["parsed_urls"] += 1

        match_status, matched_terms, language = evaluate_entity_match(
            entity_name,
            aliases,
            extra_terms,
            title,
            article_text,
            description,
            extract_domain(str(article.get("url", "") or "")),
            [],
        )
        document = build_news_document(entity_name, query, article, match_status, matched_terms, language)

        if source_name:
            source_counter[source_name] += 1

        if match_status == "relevant":
            report["relevant_count"] += 1
            documents.append(document)
        elif match_status == "possible":
            report["possible_count"] += 1
            report["possible_pages"].append(document["metadata"])
            documents.append(document)
        else:
            report["rejected_count"] += 1
            report["rejected_pages"].append(document["metadata"])

    report["source_stats"] = dict(source_counter.most_common(5))
    return documents, report


def collect_internet_mentions(search_params: Dict[str, Any]) -> Tuple[List[Document], Dict[str, Any]]:
    """Ищет и парсит интернет-упоминания юрлица."""
    entity_name = str(search_params.get("entity_name", "")).strip()
    aliases = list(search_params.get("aliases", []))
    extra_terms = list(search_params.get("extra_terms", []))
    allowed_domains = [item.lower() for item in search_params.get("allowed_domains", []) if str(item).strip()]
    blocked_domains = [item.lower() for item in search_params.get("blocked_domains", []) if str(item).strip()]
    max_results = min(max(int(search_params.get("max_results", 5)), 1), MAX_SEARCH_RESULTS)

    if not entity_name:
        raise ValueError("Укажите юридическое лицо для поиска интернет-упоминаний.")

    report: Dict[str, Any] = {
        "report_type": "internet",
        "entity_name": entity_name,
        "queries": [],
        "search_errors": [],
        "fetch_errors": [],
        "rejected_pages": [],
        "possible_pages": [],
        "found_urls": 0,
        "parsed_urls": 0,
        "relevant_count": 0,
        "possible_count": 0,
        "rejected_count": 0,
    }

    queries = build_search_queries(entity_name, aliases, extra_terms)
    report["queries"] = queries
    search_results: List[Dict[str, str]] = []

    for query in queries:
        try:
            for item in search_web(query, max_results):
                item["query"] = query
                search_results.append(item)
        except Exception as error:
            report["search_errors"].append(f"{query}: {error}")

    unique_results: List[Dict[str, str]] = []
    seen_urls: Set[str] = set()
    for item in search_results:
        url = str(item.get("url", "")).strip()
        if not is_allowed_url(url, allowed_domains, blocked_domains):
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)
        unique_results.append(item)
        if len(unique_results) >= max_results:
            break

    report["found_urls"] = len(unique_results)
    documents: List[Document] = []

    for result in unique_results:
        url = result.get("url", "").strip()
        try:
            page_payload = parse_html_page(fetch_url_text(url))
            report["parsed_urls"] += 1
        except (urllib_error.URLError, TimeoutError, ValueError) as error:
            error_message = f"{url}: {error}"
            report["fetch_errors"].append(error_message)
            fallback_document = build_internet_document(
                entity_name,
                result.get("query", ""),
                result,
                {"title": result.get("title", ""), "description": result.get("snippet", ""), "published_at": "", "text": ""},
                "possible",
                [],
                "unknown",
                fetch_error=str(error),
            )
            report["possible_pages"].append(fallback_document["metadata"])
            documents.append(fallback_document)
            report["possible_count"] += 1
            continue

        match_status, matched_terms, language = evaluate_entity_match(
            entity_name,
            aliases,
            extra_terms,
            page_payload.get("title", ""),
            page_payload.get("text", ""),
            result.get("snippet", ""),
            extract_domain(url),
            allowed_domains,
        )
        document = build_internet_document(
            entity_name,
            result.get("query", ""),
            result,
            page_payload,
            match_status,
            matched_terms,
            language,
        )

        if match_status == "relevant":
            report["relevant_count"] += 1
            documents.append(document)
        elif match_status == "possible":
            report["possible_count"] += 1
            report["possible_pages"].append(document["metadata"])
            documents.append(document)
        else:
            report["rejected_count"] += 1
            report["rejected_pages"].append(document["metadata"])

    return documents, report


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


def collect_sources(text_input: str, uploaded_files, input_mode: str, source_params: Optional[Dict[str, Any]] = None) -> Tuple[List[Document], Dict[str, Any]]:
    """Собирает список документов для анализа в едином формате."""
    if input_mode == "internet":
        return collect_internet_mentions(source_params or {})
    if input_mode == "news":
        return collect_news_mentions(source_params or {})

    if uploaded_files:
        documents: List[Document] = []
        for uploaded_file in uploaded_files:
            documents.extend(extract_documents_from_upload(uploaded_file))
        return documents, {}

    return (
        build_review_documents(
            split_reviews_from_text(text_input),
            "Текст из поля ввода",
            "text_input",
            metadata={"format": "text"},
        ),
        {},
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


def render_input_sidebar(text_input: str, uploaded_files, input_mode: str, source_params: Optional[Dict[str, Any]] = None) -> None:
    """Показывает сопроводительную панель рядом с вводом."""
    files_count = len(uploaded_files) if uploaded_files else 0
    char_count = len(text_input.strip())
    source_params = source_params or {}
    input_ready = text_input.strip() or files_count
    if input_mode in ("internet", "news"):
        input_ready = bool(str(source_params.get("entity_name", "")).strip())
    text_state = "Готов к анализу" if input_ready else "Ожидает данные"
    mode_label_map = {
        "text": "Текст и файлы",
        "internet": "Интернет-упоминания",
        "news": "Новости",
    }
    mode_label = mode_label_map.get(input_mode, "Текст и файлы")
    query_count = len(build_search_queries(
        str(source_params.get("entity_name", "")),
        list(source_params.get("aliases", [])),
        list(source_params.get("extra_terms", [])),
    )) if input_mode == "internet" else (1 if input_mode == "news" and str(source_params.get("entity_name", "")).strip() else 0)
    mode_caption = "Форматы: TXT, MD, CSV, JSON"
    if input_mode == "internet":
        mode_caption = "Поиск + парсинг URL"
    elif input_mode == "news":
        mode_caption = "NewsAPI.org + анализ статей"

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
            <div class="mini-grid" style="margin-top:0.8rem;">
                <div class="micro-card">
                    <div class="micro-card__label">Режим</div>
                    <div class="micro-card__value" style="font-size:0.95rem;">{html.escape(mode_label)}</div>
                </div>
                <div class="micro-card">
                    <div class="micro-card__label">Поисковых запросов</div>
                    <div class="micro-card__value">{query_count}</div>
                </div>
            </div>
            <div class="chip-row">
                <span class="ui-badge" style="background:{APP_PALETTE['accent_soft']};color:{APP_PALETTE['accent']};border-color:{APP_PALETTE['accent']}22;">{text_state}</span>
                <span class="ui-badge" style="background:#f8fbff;color:{APP_PALETTE['muted']};border-color:{APP_PALETTE['border']};">{mode_caption}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_internet_collection_report(collection_report: Dict[str, Any]) -> None:
    """Показывает метрики и проблемные страницы интернет-поиска."""
    if not collection_report:
        return

    report_type = str(collection_report.get("report_type", "internet"))
    title = "Интернет-источники" if report_type == "internet" else "Новостные источники"
    found_label = "Найдено URL" if report_type == "internet" else "Найдено статей"
    found_caption = "После дедупликации" if report_type == "internet" else "Получено из NewsAPI"
    parsed_label = "Распарсено" if report_type == "internet" else "Подготовлено"
    parsed_caption = "Страницы с извлеченным текстом" if report_type == "internet" else "Статей с доступным текстом"
    rejected_caption = "Нерелевантные страницы" if report_type == "internet" else "Нерелевантные статьи"

    with st.container(border=True):
        st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
        report_cols = st.columns(4)
        with report_cols[0]:
            st.markdown(render_metric_card(found_label, str(collection_report.get("found_urls", 0)), found_caption, "neutral"), unsafe_allow_html=True)
        with report_cols[1]:
            st.markdown(render_metric_card(parsed_label, str(collection_report.get("parsed_urls", 0)), parsed_caption, "positive"), unsafe_allow_html=True)
        with report_cols[2]:
            st.markdown(render_metric_card("Релевантные", str(collection_report.get("relevant_count", 0)), "Точные совпадения", "positive"), unsafe_allow_html=True)
        with report_cols[3]:
            st.markdown(render_metric_card("Отклонено", str(collection_report.get("rejected_count", 0)), rejected_caption, "negative"), unsafe_allow_html=True)

        if collection_report.get("search_errors"):
            st.markdown("**Ошибки поиска**")
            for error_message in collection_report["search_errors"]:
                st.info(error_message)

        if collection_report.get("fetch_errors"):
            st.markdown("**Ошибки получения страниц**")
            for error_message in collection_report["fetch_errors"]:
                st.info(error_message)

        source_stats = collection_report.get("source_stats", {})
        if isinstance(source_stats, dict) and source_stats:
            st.markdown("**Топ источников**")
            for source_name, count in source_stats.items():
                st.markdown(f"- {source_name}: {count}")

        if collection_report.get("possible_pages"):
            st.markdown("**Сомнительные совпадения**")
            for page in collection_report["possible_pages"][:5]:
                url = str(page.get("url", "")).strip()
                page_domain = str(page.get("source_name", "")).strip() or str(page.get("domain", "")).strip() or "источник"
                matched_terms = ", ".join(page.get("matched_terms", [])) or "без явных маркеров"
                line = f"- {page_domain}: {matched_terms}"
                if url:
                    st.markdown(f"{line}  \n[{url}]({url})")
                else:
                    st.markdown(line)

        if collection_report.get("rejected_pages"):
            st.markdown("**Отклоненные страницы**")
            for page in collection_report["rejected_pages"][:5]:
                url = str(page.get("url", "")).strip()
                page_domain = str(page.get("source_name", "")).strip() or str(page.get("domain", "")).strip() or "источник"
                title = str(page.get("search_title", "")).strip() or "Без заголовка"
                if url:
                    st.markdown(f"- {page_domain}: {html.escape(title)}  \n[{url}]({url})", unsafe_allow_html=False)
                else:
                    st.markdown(f"- {page_domain}: {title}")


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


def render_summary_dashboard(analyses: List[AnalysisResult], collection_report: Optional[Dict[str, Any]] = None) -> None:
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

    if collection_report:
        render_internet_collection_report(collection_report)


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
    status_map = {
        "relevant": "Точное совпадение",
        "possible": "Сомнительное совпадение",
        "rejected": "Отклонено",
    }

    with st.container(border=True):
        header_col, badge_col = st.columns([0.82, 0.18], gap="small")
        with header_col:
            st.markdown(
                f"<div class='section-title' style='margin-bottom:0.2rem;'>{html.escape(source_name)}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='section-text'>Формат: {html.escape(str(metadata.get('format', source_type)))}</div>",
                unsafe_allow_html=True,
            )
            if source_type in ("internet_mention", "news_mention"):
                url = str(metadata.get("url", "")).strip()
                domain = str(metadata.get("source_name", "")).strip() or str(metadata.get("domain", "")).strip() or "Неизвестный домен"
                published_at = str(metadata.get("published_at", "")).strip() or "Дата не найдена"
                match_status = str(metadata.get("match_status", "")).strip() or "unknown"
                matched_terms = ", ".join(metadata.get("matched_terms", [])) or "без явных совпадений"
                st.caption(f"Источник: {domain}")
                st.caption(f"Дата: {published_at}")
                st.caption(f"Статус: {status_map.get(match_status, match_status)}")
                st.caption(f"Совпадения: {matched_terms}")
                if url:
                    st.markdown(f"[Открыть источник]({url})")
        with badge_col:
            st.markdown(render_badge(sentiment_ui["label"], bucket), unsafe_allow_html=True)

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

    text_input = ""
    uploaded_files = []
    source_params: Dict[str, Any] = {}

    input_col, support_col = st.columns([1.22, 0.78], gap="medium")
    with input_col:
        with st.container(border=True):
            st.markdown("<div class='section-title'>Ввод данных</div>", unsafe_allow_html=True)
            input_mode = st.radio(
                "Источник данных",
                options=["Текст и файлы", "Интернет-упоминания", "Новости"],
                horizontal=True,
            )
            if input_mode == "Текст и файлы":
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
            elif input_mode == "Интернет-упоминания":
                entity_name = st.text_input(
                    "Юридическое лицо",
                    placeholder="Например: ООО Ромашка",
                    help="Официальное название компании или ключевой вариант, который должен находиться на странице.",
                )
                aliases_raw = st.text_area(
                    "Алиасы и варианты названия",
                    height=90,
                    placeholder="ООО Ромашка, Ромашка, Romashka LLC",
                    help="Запятая или новая строка между вариантами.",
                )
                extra_terms_raw = st.text_input(
                    "Дополнительные маркеры",
                    placeholder="Москва, кофейня, inn 7701...",
                    help="Необязательные слова, которые усиливают уверенность в совпадении.",
                )
                allowed_domains_raw = st.text_input(
                    "Разрешенные домены",
                    placeholder="vedomosti.ru, 2gis.ru",
                    help="Если заполнено, парсинг будет ограничен только этими доменами.",
                )
                blocked_domains_raw = st.text_input(
                    "Исключаемые домены",
                    placeholder="youtube.com, vk.com",
                    help="Необязательный список доменов, которые нужно пропускать.",
                )
                max_results = st.number_input(
                    "Максимум результатов",
                    min_value=1,
                    max_value=MAX_SEARCH_RESULTS,
                    value=5,
                    step=1,
                    help="Сколько URL брать в обработку после дедупликации.",
                )
                source_params = {
                    "entity_name": entity_name,
                    "aliases": split_comma_lines(aliases_raw),
                    "extra_terms": split_comma_lines(extra_terms_raw),
                    "allowed_domains": split_comma_lines(allowed_domains_raw),
                    "blocked_domains": split_comma_lines(blocked_domains_raw),
                    "max_results": int(max_results),
                }
            else:
                entity_name = st.text_input(
                    "Юридическое лицо",
                    placeholder="Например: ЛАНИТ",
                    help="Основное название компании для поиска по новостям.",
                )
                aliases_raw = st.text_area(
                    "Алиасы и варианты названия",
                    height=90,
                    placeholder="ЛАНИТ, ГК ЛАНИТ, LANIT",
                    help="Запятая или новая строка между вариантами.",
                )
                extra_terms_raw = st.text_input(
                    "Дополнительные маркеры",
                    placeholder="ИТ, интегратор, цифровизация",
                    help="Слова для усиления релевантности статьи на этапе матчинга.",
                )
                language = st.selectbox(
                    "Язык новостей",
                    options=["ru", "en"],
                    index=0,
                    help="Фильтр языка на уровне NewsAPI.",
                )
                date_cols = st.columns(2)
                with date_cols[0]:
                    date_from = st.date_input("С даты", value=None, format="YYYY-MM-DD")
                with date_cols[1]:
                    date_to = st.date_input("По дату", value=None, format="YYYY-MM-DD")
                allowed_sources_raw = st.text_input(
                    "Разрешенные источники",
                    placeholder="vedomosti, forbes, cnews.ru",
                    help="Необязательный список source id или source name из NewsAPI.",
                )
                max_results = st.number_input(
                    "Максимум результатов",
                    min_value=1,
                    max_value=NEWSAPI_MAX_RESULTS,
                    value=10,
                    step=1,
                    help="Сколько статей запросить у NewsAPI.",
                )
                st.caption("Для режима новостей требуется переменная окружения NEWSAPI_API_KEY.")
                source_params = {
                    "entity_name": entity_name,
                    "aliases": split_comma_lines(aliases_raw),
                    "extra_terms": split_comma_lines(extra_terms_raw),
                    "language": language,
                    "date_from": date_from.isoformat() if date_from else "",
                    "date_to": date_to.isoformat() if date_to else "",
                    "allowed_sources": split_comma_lines(allowed_sources_raw),
                    "max_results": int(max_results),
                }
            analyze_trigger = st.button("Запустить анализ")

    with support_col:
        st.markdown("<div class='sidebar-stack'>", unsafe_allow_html=True)
        render_input_sidebar(
            text_input,
            uploaded_files,
            "internet" if input_mode == "Интернет-упоминания" else ("news" if input_mode == "Новости" else "text"),
            source_params,
        )
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
            sources, collection_report = collect_sources(
                text_input,
                uploaded_files,
                "internet" if input_mode == "Интернет-упоминания" else ("news" if input_mode == "Новости" else "text"),
                source_params,
            )
            if not sources or all(not str(document.get("text", "")).strip() for document in sources):
                if input_mode == "Интернет-упоминания":
                    st.warning("Не удалось найти релевантные интернет-упоминания. Проверьте название компании, доступ в интернет и ограничения по доменам.")
                    if collection_report:
                        render_internet_collection_report(collection_report)
                elif input_mode == "Новости":
                    st.warning("Не удалось найти релевантные новости. Проверьте название компании, NEWSAPI_API_KEY и фильтры по датам или источникам.")
                    if collection_report:
                        render_internet_collection_report(collection_report)
                else:
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
                    render_summary_dashboard(analyses, collection_report)
                with result_tabs[1]:
                    st.markdown(
                        "<div class='section-title'>Отзывы и упоминания</div>",
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


