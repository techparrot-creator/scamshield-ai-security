from scamshield.query_planner import infer_retrieval_hints, should_expand_query


def test_pakistan_otp_hints():
    hints = infer_retrieval_hints("JazzCash account ka OTP share karo")
    assert hints["country"] == "Pakistan"
    assert "otp_account_takeover" in hints["topics"]


def test_expand_on_low_confidence():
    assert should_expand_query(0.2, 4, 0.43)
    assert not should_expand_query(0.8, 4, 0.43)
