# System Architecture
Download mermaid preview extension for markdown files to be able to see this file correctly!

```mermaid
flowchart TD
    A[pg_stat_statements<br/>poll every 5s] --> B[Feature extraction<br/>8 lexical + 12 plan = 20 features]
    B --> C{Multi-task neural classifier<br/>shared trunk 20→64→32}
    C -->|classification head| D{p_slow ≥ 0.5?}
    C -->|regression head| H[predicted_ms]
    D -->|no| Z[skip query]
    D -->|yes| E[Embedding lookup<br/>all-MiniLM-L6-v2<br/>cosine similarity]
    E -->|similar fix found| F
    E -->|no match| F
    F[Fine-tuned LLM<br/>defog/sqlcoder-7b + QLoRA adapter<br/>4-bit NF4 quantization]
    F --> G[3 candidate SQL rewrites<br/>beam search + temperature sampling]
    G --> I[Re-score each candidate<br/>via regression head]
    H --> I
    I --> J[Rank by predicted speedup<br/>predicted_ms original − predicted_ms candidate]
    J --> K[Verify top candidate<br/>cold + warm EXPLAIN ANALYZE]
    K -->|speedup confirmed| L[Store in embedding history<br/>slow query + fix + speedup_ms]
    K --> M[Log structured event<br/>dashboard/query_log.json]
    M --> N[Flask dashboard<br/>live charts + query cards]

    style C fill:#f0e0ff,color:#000
    style F fill:#e0f0ff,color:#000
    style J fill:#fff0e0,color:#000
    style K fill:#e0ffe0,color:#000
    style L fill:#e0ffe0,color:#000
    style N fill:#e8e8ff,color:#000
```

## Component summary

| Stage | Component | File |
|---|---|---|
| 1. Detection | pg_stat_statements poll | `monitor/monitor.py` |
| 2. Feature extraction | Lexical + EXPLAIN ANALYZE | `classifier/inference.py` |
| 3. Classification | Multi-task PyTorch model | `classifier/classifier.py` |
| 4. Memory lookup | Sentence-transformer similarity | `embeddings/embeddings.py` |
| 5. Rewrite generation | sqlcoder-7b QLoRA fine-tune | `finetune/llm_optimizer.py` |
| 6. Ranking | Regression head re-scoring | `classifier/inference.py` |
| 7. Verification | Cold + warm DB timing | `monitor/monitor.py` |
| 8. Storage | Embedding history + JSON log | `embeddings/embeddings.py` |
| 9. Visualisation | Flask + Chart.js dashboard | `dashboard/app.py` |
