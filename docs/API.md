# API reference

```bash
# Query (dev — body user_id/role when AUTH_ALLOW_BODY_FALLBACK=true)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Which customers ordered the most?", "user_id": "regular_001", "role": "regular_office"}'

# Query (production — Google ID token)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GOOGLE_ID_TOKEN" \
  -d '{"question": "List all table of contents for the Go.Data document."}'

# Upload PDF (admin JWT required)
curl -X POST http://localhost:8000/ingest/unstructured \
  -H "Authorization: Bearer $GOOGLE_ID_TOKEN" \
  -F "file=@doc.pdf" -F "doc_key=my-doc"

# Job status (admin JWT required)
curl -H "Authorization: Bearer $GOOGLE_ID_TOKEN" \
  http://localhost:8000/ingest/jobs/{job_id}
curl -H "Authorization: Bearer $GOOGLE_ID_TOKEN" \
  "http://localhost:8000/ingest/jobs?limit=20"
curl -H "Authorization: Bearer $GOOGLE_ID_TOKEN" \
  http://localhost:8000/ingest/queue/status

# Audit trail (admin JWT required)
curl -H "Authorization: Bearer $GOOGLE_ID_TOKEN" \
  "http://localhost:8000/audit/events?limit=20&event_type=access_denied"

# Auth helpers
curl http://localhost:8000/auth/config
curl -H "Authorization: Bearer $GOOGLE_ID_TOKEN" http://localhost:8000/auth/me

# Health / active models
curl http://localhost:8000/health
curl http://localhost:8000/config/models

# Feedback loop (requires RETRIEVAL_FEEDBACK_ENABLED=true)
curl -X POST http://localhost:8000/feedback/outcome \
  -H "Content-Type: application/json" \
  -d '{"request_id": "YOUR_REQUEST_ID", "passed": true}'
curl "http://localhost:8000/feedback/stats?question=monthly%20order%20count%20in%201997"
```

Response fields from `/ingest/jobs/{id}`: `status`, `dispatch` (`worker` / `background_task`), `logical_doc_id`, `revision_id`, `version_number`, `skipped_duplicate`, `logs[]`, `error`.

Full interactive reference: `GET /docs` (Swagger UI).
