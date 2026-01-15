Production readiness checklist and changes applied

Summary of changes made:
- Implemented sentence-aware chunking and better overlap handling in `src/vectordb.py`.
- Added query normalization, optional TF-IDF expansion, metadata filtering, and `evaluate_retrieval()` in `src/vectordb.py`.
- Added batched embedding and ingestion to ChromaDB and introduced `EMBED_BATCH_SIZE` env var for tuning.
- Replaced ad-hoc prints with structured logging in `src/app.py` and `src/vectordb.py`.
- Added `scripts/run_eval.py` for quick retrieval evaluations and a smoke test under `src/smoke_test.py` (existing).
- Added a `Dockerfile` and `.dockerignore` for containerized runs.

Recommended next steps for full production hardening:

- Configuration & Secrets:
  - Use a secrets manager (Vault, AWS Secrets Manager, GCP Secret Manager) for API keys.
  - Add typed config (e.g., `pydantic.BaseSettings`) to centralize configuration.

- Observability:
  - Integrate structured logging and JSON output format.
  - Add OpenTelemetry tracing (SDK already in requirements) and export to your collector.
  - Emit metrics (Prometheus) for request counts, latencies, and embedding throughput.

- Reliability & Performance:
  - Run embeddings in async worker or queue (Celery/RQ) for large ingestions.
  - Add retries and exponential backoff around external calls (ChromaDB, LLM APIs).
  - Use chunking tuned for your LLM context window + tokenization-based chunk sizing.

- Security & Compliance:
  - Do not persist raw API keys in `.env` in production.
  - Add input validation and size limits to avoid expensive requests.

- CI/CD & Testing:
  - Add unit tests for `evaluate.py` and `vectordb` functions (use small mocked embeddings).
  - Add a GitHub Actions workflow to run lint, tests, and the smoke test.
  - Build and scan Docker images for vulnerabilities.

- Deployment:
  - Provide a Kubernetes manifest or Helm chart with resource limits, liveness/readiness probes, and autoscaling rules.
  - Use persistent storage or managed vector DB service for ChromaDB data.

How to run locally:

- Activate your virtualenv and install requirements:

  ```powershell
  & rag_env\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```

- Run the smoke test:

  ```powershell
  python -m src.smoke_test
  ```

- Run the evaluation script (quick example):

  ```powershell
  python scripts/run_eval.py
  ```

If you want, I can:
- Add a `pyproject.toml` and make the package importable as a module.
- Add GitHub Actions CI and a simple Kubernetes manifest.
- Add unit tests and a small benchmark harness for chunking/embedding throughput.

Which of these should I do next?