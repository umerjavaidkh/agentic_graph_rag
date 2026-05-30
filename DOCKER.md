# Docker demo

## Quick start (slim ~1 GB app)

```bash
cp .env.example .env   # set OPENAI_API_KEY
docker compose up --build
```

| URL | Address |
|-----|---------|
| Chat | http://localhost:8000/ |
| Neo4j Browser | http://localhost:**17474** (not 7474) |
| Neo4j login | `neo4j` / `password123` |

**Why 17474?** Avoids conflict if you already run Neo4j on ports 7474 / 7687.

First build ~3–5 min (slim). Northwind loads on first boot.

---

## Image sizes

| Image | Size | PDF upload |
|-------|------|------------|
| **Default** (`Dockerfile`) | ~**0.5–1 GB** | No |
| **Full** (`Dockerfile.full` + Docling/PyTorch) | ~**8–10 GB** | Yes |

### PDF ingest (large image only)

```bash
docker compose -f docker-compose.yml -f docker-compose.full.yml up --build
```

---

## Neo4j already running locally?

Use your existing DB on `7687` (don’t start bundled Neo4j):

```bash
# .env → NEO4J_PASSWORD=your-password
docker compose -f docker-compose.yml -f docker-compose.external-neo4j.yml up --build --scale neo4j=0
```

Load Northwind once via Neo4j Browser if empty (`docker/northwind-docker.cypher`).

---

## Stop / reset

```bash
docker compose down
docker compose down -v   # wipe DB + assets
```

---

## Remove old 10 GB image

```bash
docker compose down
docker rmi experimental_practice-app:latest
docker compose up --build
```
