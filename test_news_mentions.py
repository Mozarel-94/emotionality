import importlib
import sys
import types
import unittest
from unittest.mock import patch


def install_test_stubs() -> None:
    if "streamlit" not in sys.modules:
        streamlit_stub = types.ModuleType("streamlit")

        def cache_resource(func=None, **_kwargs):
            if func is None:
                return lambda inner: inner
            return func

        streamlit_stub.cache_resource = cache_resource
        streamlit_stub.markdown = lambda *args, **kwargs: None
        streamlit_stub.info = lambda *args, **kwargs: None
        streamlit_stub.warning = lambda *args, **kwargs: None
        streamlit_stub.error = lambda *args, **kwargs: None
        streamlit_stub.write = lambda *args, **kwargs: None
        streamlit_stub.caption = lambda *args, **kwargs: None
        streamlit_stub.progress = lambda *args, **kwargs: types.SimpleNamespace(progress=lambda *a, **k: None, empty=lambda: None)
        streamlit_stub.container = lambda *args, **kwargs: types.SimpleNamespace(__enter__=lambda self: self, __exit__=lambda self, exc_type, exc, tb: False)
        streamlit_stub.columns = lambda *args, **kwargs: []
        streamlit_stub.tabs = lambda labels: [types.SimpleNamespace(__enter__=lambda self: self, __exit__=lambda self, exc_type, exc, tb: False) for _ in labels]
        streamlit_stub.spinner = lambda *args, **kwargs: types.SimpleNamespace(__enter__=lambda self: self, __exit__=lambda self, exc_type, exc, tb: False)
        streamlit_stub.set_page_config = lambda *args, **kwargs: None
        sys.modules["streamlit"] = streamlit_stub

    if "altair" not in sys.modules:
        sys.modules["altair"] = types.ModuleType("altair")

    if "transformers" not in sys.modules:
        transformers_stub = types.ModuleType("transformers")
        transformers_stub.pipeline = lambda *args, **kwargs: None
        sys.modules["transformers"] = transformers_stub


install_test_stubs()
emotion = importlib.import_module("emotion")


class NewsMentionsTests(unittest.TestCase):
    def test_build_newsapi_query_with_aliases(self) -> None:
        query = emotion.build_newsapi_query("Систематика консалтинг", ["Систематика", "Systematica"])
        self.assertEqual(query, '"Систематика консалтинг" OR ("Систематика" OR "Systematica")')

    def test_build_newsapi_query_without_aliases(self) -> None:
        query = emotion.build_newsapi_query("ЛАНИТ", [])
        self.assertEqual(query, '"ЛАНИТ"')

    def test_fetch_newsapi_articles_raises_for_missing_key(self) -> None:
        with patch.object(emotion, "load_newsapi_config", return_value={"api_key": "", "base_url": "https://newsapi.org/v2/everything"}):
            with self.assertRaises(ValueError):
                emotion.fetch_newsapi_articles({"entity_name": "ЛАНИТ", "aliases": []})

    def test_fetch_newsapi_articles_success(self) -> None:
        payload = {"status": "ok", "articles": [{"title": "Lanit news"}]}
        with patch.object(emotion, "load_newsapi_config", return_value={"api_key": "secret", "base_url": "https://newsapi.org/v2/everything"}), patch.object(emotion, "fetch_json", return_value=payload) as fetch_json_mock:
            result = emotion.fetch_newsapi_articles(
                {
                    "entity_name": "ЛАНИТ",
                    "aliases": ["LANIT"],
                    "language": "en",
                    "date_from": "2026-03-01",
                    "date_to": "2026-03-18",
                    "max_results": 5,
                }
            )

        self.assertEqual(result, payload)
        called_url = fetch_json_mock.call_args.args[0]
        self.assertIn("q=%22%D0%9B%D0%90%D0%9D%D0%98%D0%A2%22+OR+%28%22LANIT%22%29", called_url)
        self.assertIn("language=en", called_url)
        self.assertIn("from=2026-03-01", called_url)
        self.assertIn("to=2026-03-18", called_url)

    def test_fetch_newsapi_articles_raises_on_api_error(self) -> None:
        with patch.object(emotion, "load_newsapi_config", return_value={"api_key": "secret", "base_url": "https://newsapi.org/v2/everything"}), patch.object(emotion, "fetch_json", return_value={"status": "error", "message": "apiKeyInvalid"}):
            with self.assertRaises(ValueError):
                emotion.fetch_newsapi_articles({"entity_name": "ЛАНИТ", "aliases": []})

    def test_build_news_document_normalizes_metadata(self) -> None:
        article = {
            "source": {"id": "forbes", "name": "Forbes"},
            "author": "Reporter",
            "title": "Lanit expands",
            "description": "New digital project",
            "content": "Lanit launched a new product.",
            "url": "https://example.com/lanit-story",
            "publishedAt": "2026-03-18T10:00:00Z",
        }
        document = emotion.build_news_document("ЛАНИТ", '"ЛАНИТ"', article, "relevant", ["ЛАНИТ"], "en")

        self.assertEqual(document["source_type"], "news_mention")
        self.assertIn("Lanit expands", document["text"])
        self.assertEqual(document["metadata"]["source_name"], "Forbes")
        self.assertEqual(document["metadata"]["published_at"], "2026-03-18T10:00:00Z")
        self.assertEqual(document["metadata"]["match_status"], "relevant")

    def test_collect_news_mentions_filters_relevance_and_sources(self) -> None:
        articles = [
            {
                "source": {"id": "cnews", "name": "CNews"},
                "author": "Editor",
                "title": "Систематика консалтинг усилила ИТ-направление",
                "description": "Компания Систематика консалтинг расширяет практику интеграции.",
                "content": "Систематика консалтинг объявила о развитии ИТ-услуг и консалтинга.",
                "url": "https://example.com/systematica-news",
                "publishedAt": "2026-03-18T10:00:00Z",
            },
            {
                "source": {"id": "other", "name": "Other News"},
                "author": "Reporter",
                "title": "Generic market update",
                "description": "No target company here.",
                "content": "This article is not about the requested legal entity.",
                "url": "https://example.com/other",
                "publishedAt": "2026-03-17T10:00:00Z",
            },
        ]
        with patch.object(emotion, "fetch_newsapi_articles", return_value={"status": "ok", "articles": articles}):
            documents, report = emotion.collect_news_mentions(
                {
                    "entity_name": "Систематика консалтинг",
                    "aliases": ["Систематика"],
                    "extra_terms": ["ИТ"],
                    "language": "ru",
                    "date_from": "2026-03-01",
                    "date_to": "2026-03-18",
                    "allowed_sources": ["cnews"],
                    "max_results": 10,
                }
            )

        self.assertEqual(len(documents), 1)
        self.assertEqual(report["found_urls"], 2)
        self.assertEqual(report["parsed_urls"], 1)
        self.assertEqual(report["relevant_count"], 1)
        self.assertEqual(report["source_stats"], {"CNews": 1})
        self.assertEqual(documents[0]["metadata"]["source_name"], "CNews")

    def test_collect_news_mentions_handles_multiple_entities(self) -> None:
        cases = [
            ("Систематика консалтинг", ["Систематика"], "Систематика консалтинг выиграла новый ИТ-тендер"),
            ("систематика", ["Систематика консалтинг"], "Систематика расширяет команду разработки"),
            ("ланит", ["ГК ЛАНИТ"], "ЛАНИТ запустил новый центр разработки"),
        ]
        for entity_name, aliases, title in cases:
            with self.subTest(entity=entity_name):
                with patch.object(
                    emotion,
                    "fetch_newsapi_articles",
                    return_value={
                        "status": "ok",
                        "articles": [
                            {
                                "source": {"id": "news", "name": "News Outlet"},
                                "author": "Reporter",
                                "title": title,
                                "description": title,
                                "content": title + " и усилил присутствие на рынке.",
                                "url": "https://example.com/article",
                                "publishedAt": "2026-03-18T10:00:00Z",
                            }
                        ],
                    },
                ):
                    documents, report = emotion.collect_news_mentions(
                        {
                            "entity_name": entity_name,
                            "aliases": aliases,
                            "extra_terms": ["ИТ"],
                            "language": "ru",
                            "date_from": "",
                            "date_to": "",
                            "allowed_sources": [],
                            "max_results": 10,
                        }
                    )

                self.assertEqual(len(documents), 1)
                self.assertEqual(documents[0]["source_type"], "news_mention")
                self.assertEqual(report["relevant_count"], 1)


if __name__ == "__main__":
    unittest.main()
