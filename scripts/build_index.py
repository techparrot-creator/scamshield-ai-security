from scamshield.retrieval import rag_health, search_once

print("ScamShield RAG diagnostics:", rag_health())
results = search_once("bank OTP phishing WhatsApp verification code", top_k=3, hints={"country": "Pakistan", "topics": ["otp_account_takeover", "phishing"]})
for result in results:
    print(result["retrieval_score"], result["title"], "->", result["section_title"])
