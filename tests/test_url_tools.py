from scamshield.url_tools import analyze_url, extract_urls


def test_extract_urls():
    text = "See https://example.com/a and http://1.2.3.4/login now."
    assert extract_urls(text) == ["https://example.com/a", "http://1.2.3.4/login"]


def test_ip_url_is_flagged():
    result = analyze_url("http://1.2.3.4/login")
    assert result["heuristic_score"] >= 25
    assert any("raw IP" in signal for signal in result["signals"])
