from rag.text import normalize, tokenize


def test_normalize_unifies_alef_and_strips_diacritics():
    assert normalize("أَحْمَد") == normalize("احمد")
    assert normalize("إجازة") == "اجازه"


def test_tokenize_drops_article_and_stopwords():
    tokens = tokenize("ما هي مدة الإجازة السنوية؟")
    assert "اجازه" in tokens and "سنويه" in tokens
    assert "ما" not in tokens and "هي" not in tokens


def test_tokenize_english():
    assert tokenize("The refund policy for annual plans") == ["refund", "policy", "annual", "plans"]
