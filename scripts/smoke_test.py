from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def main() -> int:
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        print("BLOCKED: set GOOGLE_API_KEY in .env before running the end-to-end smoke test.")
        return 2

    from scamshield.graph import graph

    result = graph.invoke(
        {
            "raw_input": (
                "Main bank security team se hun. Apna OTP foran bhejo warna account block ho jayega. "
                "USER INTERACTION STATUS: Clicked suspicious link: no; Sent money: no; Shared secret: no."
            ),
            "input_kind": "text",
            "preferred_language": "Roman Urdu",
        }
    )
    assessment = result.get("assessment", {})
    print("Case:", result.get("case_id"))
    print("Risk:", assessment.get("risk_level"), assessment.get("risk_score"))
    print("Type:", assessment.get("scam_type"))
    print("RAG confidence:", result.get("retrieval_confidence"))
    print("Sources:", len(result.get("retrieved_guidance", [])))

    required = ["case_id", "assessment", "response_markdown", "report_markdown"]
    missing = [key for key in required if not result.get(key)]
    if missing:
        print("FAIL: missing output fields:", ", ".join(missing))
        return 1
    if not result.get("retrieved_guidance"):
        print("FAIL: no RAG guidance was retrieved")
        return 1
    print("PASS: end-to-end LangGraph + RAG + Gemini smoke test completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
