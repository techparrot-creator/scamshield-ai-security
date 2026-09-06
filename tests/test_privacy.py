from scamshield.privacy import redact_sensitive_text


def test_redacts_otp_and_cnic():
    text = "OTP: 123456 CNIC 12345-1234567-1"
    redacted = redact_sensitive_text(text)
    assert "123456" not in redacted
    assert "12345-1234567-1" not in redacted


def test_masks_email():
    assert "alice@example.com" not in redact_sensitive_text("email alice@example.com")
