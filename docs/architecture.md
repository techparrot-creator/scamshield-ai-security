# ScamShield AI architecture

```mermaid
flowchart TD
    A[Gradio: text / link / screenshot / voice] --> B[Media evidence extraction]
    B --> C[LangGraph normalize + case ID]
    C --> D[URL structural analysis]
    D --> E[Query planner + metadata hints]
    E --> F[Advanced RAG]
    F --> F1[Hugging Face multilingual embeddings]
    F --> F2[ChromaDB semantic search]
    F --> F3[BM25 keyword search]
    F2 --> F4[MMR diversity]
    F1 --> F4
    F3 --> F5[RRF fusion]
    F4 --> F5
    F5 --> F6[Metadata boost + score threshold]
    F6 --> G{Retrieval confidence weak?}
    G -- yes --> H[Gemini query expansion]
    H --> I[Multi-query retrieval + fusion]
    G -- no --> J[Parent-context retrieval]
    I --> J
    J --> K[Gemini structured scam assessment]
    K --> L[Risk + reasons + recovery + reporting]
    L --> M[Gradio + privacy-redacted case report]
```

## Why this is agentic

LangGraph controls a stateful, inspectable workflow rather than using one giant prompt. Retrieval can branch into multi-query expansion only when the initial retrieval is weak. This keeps normal requests fast while improving difficult Urdu/Roman-Urdu or ambiguous cases.

## RAG design

- Child sections are indexed for precise matching.
- Parent summaries restore broader context after retrieval.
- Hugging Face multilingual embeddings handle semantic meaning across English, Urdu, and Roman Urdu.
- BM25 catches exact indicators such as OTP, CNIC, WhatsApp, JazzCash, and specific organization names.
- MMR reduces repetitive semantic results.
- Reciprocal Rank Fusion combines semantic and keyword rankings.
- Metadata hints boost Pakistan-specific and scam-type-specific sources.
- A confidence gate triggers multi-query expansion only when needed.
