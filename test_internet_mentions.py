import importlib
import sys
import types
import unittest
from unittest.mock import patch


def install_test_stubs() -> None:
    """Подменяет тяжелые внешние зависимости легкими стабами для unit-тестов."""
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


def build_page_html(title: str, body: str, description: str = "") -> str:
    return f"""
    <html>
      <head>
        <title>{title}</title>
        <meta name="description" content="{description}">
      </head>
      <body>
        <main>
          <h1>{title}</h1>
          <p>{body}</p>
        </main>
      </body>
    </html>
    """


class InternetMentionsTests(unittest.TestCase):
    def test_detect_supported_language_accepts_russian_and_english_only(self) -> None:
        self.assertEqual(emotion.detect_supported_language("Систематика консалтинг развивает ИТ-услуги"), "ru")
        self.assertEqual(emotion.detect_supported_language("Lanit develops digital integration services"), "en")
        self.assertEqual(emotion.detect_supported_language("whatsapp 被限制了怎么办呢"), "other")

    def test_parse_bing_html_results_extracts_redirect_url(self) -> None:
        html_text = """
        <html><body>
          <li class="b_algo">
            <h2><a href="https://www.bing.com/ck/a?!&amp;&amp;u=a1aHR0cHM6Ly9leGFtcGxlLmNvbS9zeXN0ZW1hdGljYQ">Систематика Консалтинг</a></h2>
            <div class="b_caption"><p>ИТ-компания и цифровой интегратор</p></div>
          </li>
        </body></html>
        """

        results = emotion.parse_bing_html_results(html_text)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["url"], "https://example.com/systematica")
        self.assertEqual(results[0]["title"], "Систематика Консалтинг")

    def test_parse_bing_rss_results_extracts_items(self) -> None:
        xml_text = """<?xml version="1.0" encoding="utf-8"?>
        <rss version="2.0">
          <channel>
            <item>
              <title>Систематика консалтинг</title>
              <link>https://example.com/systematica</link>
              <description>ИТ-компания и интегратор</description>
            </item>
          </channel>
        </rss>
        """

        results = emotion.parse_bing_rss_results(xml_text)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["url"], "https://example.com/systematica")
        self.assertEqual(results[0]["source"], "example.com")

    def test_search_web_auto_falls_back_when_duckduckgo_is_blocked(self) -> None:
        with patch.object(emotion, "search_web_bing_html", return_value=[{"title": "ЛАНИТ", "url": "https://example.com/lanit", "snippet": "новость", "source": "example.com"}]), patch.object(emotion, "search_web_bing_rss", return_value=[]), patch.object(emotion, "search_web_duckduckgo", return_value=[]), patch.object(emotion, "load_search_config", return_value={"provider": "auto", "api_key": "", "base_url": ""}):
            results = emotion.search_web("ланит", 5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["url"], "https://example.com/lanit")

    def test_parse_duckduckgo_html_results_skips_search_engine_self_links(self) -> None:
        html_text = """
        <html><body>
          <div class="result">
            <a class="result__a" href="https://duckduckgo.com/">here</a>
            <div class="result__snippet">служебная ссылка поисковика</div>
          </div>
          <div class="result">
            <a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fsystematica">Систематика Консалтинг</a>
            <div class="result__snippet">ИТ-компания и цифровой интегратор</div>
          </div>
        </body></html>
        """

        results = emotion.parse_duckduckgo_html_results(html_text)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["url"], "https://example.com/systematica")

    def test_parse_duckduckgo_html_results_keeps_title_snippet_and_redirect_url(self) -> None:
        html_text = """
        <html><body>
          <div class="result">
            <a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fsystematica">Систематика Консалтинг</a>
            <div class="result__extras__url">example.com</div>
            <div class="result__snippet">ИТ-компания и цифровой интегратор</div>
          </div>
        </body></html>
        """

        results = emotion.parse_duckduckgo_html_results(html_text)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Систематика Консалтинг")
        self.assertEqual(results[0]["url"], "https://example.com/systematica")
        self.assertEqual(results[0]["snippet"], "ИТ-компания и цифровой интегратор")
        self.assertEqual(results[0]["source"], "example.com")

    def test_collect_internet_mentions_returns_relevant_documents_for_multiple_entities(self) -> None:
        cases = [
            {
                "entity_name": "Систематика консалтинг",
                "aliases": ["Систематика"],
                "url": "https://systematica.example/news",
                "title": "Систематика консалтинг запустила новый ИТ-проект",
                "body": "Систематика консалтинг расширила команду и усилила направление системной интеграции.",
            },
            {
                "entity_name": "систематика",
                "aliases": ["Систематика консалтинг"],
                "url": "https://systematica.example/about",
                "title": "Систематика развивает консалтинг и ИТ-услуги",
                "body": "Компания Систематика усиливает консалтинговое направление и развивает собственные ИТ-сервисы.",
            },
            {
                "entity_name": "ланит",
                "aliases": ["ГК ЛАНИТ"],
                "url": "https://lanit.example/press",
                "title": "ЛАНИТ открыл новый центр разработки",
                "body": "Группа компаний ЛАНИТ расширяет присутствие на рынке и запускает новые технологические инициативы.",
            },
        ]

        for case in cases:
            with self.subTest(entity=case["entity_name"]):
                def fake_search_web(query: str, max_results: int):
                    return [
                        {
                            "title": case["title"],
                            "url": case["url"],
                            "snippet": case["body"],
                            "source": emotion.extract_domain(case["url"]),
                            "query": query,
                        }
                    ][:max_results]

                def fake_fetch_url_text(url: str):
                    self.assertEqual(url, case["url"])
                    return build_page_html(case["title"], case["body"], case["body"])

                with patch.object(emotion, "search_web", side_effect=fake_search_web), patch.object(emotion, "fetch_url_text", side_effect=fake_fetch_url_text):
                    documents, report = emotion.collect_internet_mentions(
                        {
                            "entity_name": case["entity_name"],
                            "aliases": case["aliases"],
                            "extra_terms": ["ИТ-компания"],
                            "allowed_domains": [],
                            "blocked_domains": [],
                            "max_results": 3,
                        }
                    )

                self.assertGreaterEqual(len(documents), 1)
                self.assertGreaterEqual(report["relevant_count"], 1)
                self.assertEqual(report["found_urls"], 1)
                self.assertEqual(report["parsed_urls"], 1)
                self.assertFalse(report["search_errors"])
                self.assertFalse(report["fetch_errors"])
                self.assertEqual(documents[0]["source_type"], "internet_mention")
                self.assertEqual(documents[0]["metadata"]["match_status"], "relevant")
                self.assertTrue(documents[0]["metadata"]["matched_terms"])

    def test_collect_internet_mentions_rejects_non_ru_en_page(self) -> None:
        with patch.object(
            emotion,
            "search_web",
            return_value=[
                {
                    "title": "whatsapp 被限制了怎么办呢",
                    "url": "https://example.com/foreign",
                    "snippet": "whatsapp 被限制了怎么办呢",
                    "source": "example.com",
                    "query": "systematica",
                }
            ],
        ), patch.object(
            emotion,
            "fetch_url_text",
            return_value=build_page_html(
                "whatsapp 被限制了怎么办呢",
                "whatsapp 被限制了怎么办呢 还能找回吗",
                "whatsapp 被限制了怎么办呢",
            ),
        ):
            documents, report = emotion.collect_internet_mentions(
                {
                    "entity_name": "Систематика консалтинг",
                    "aliases": ["Систематика"],
                    "extra_terms": [],
                    "allowed_domains": [],
                    "blocked_domains": [],
                    "max_results": 3,
                }
            )

        self.assertEqual(documents, [])
        self.assertEqual(report["rejected_count"], 1)
        self.assertEqual(report["rejected_pages"][0]["language"], "other")


if __name__ == "__main__":
    unittest.main()
