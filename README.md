# Interview-Platform-Fastapi
On Ubuntu, you can start PostgreSQL with:

sudo systemctl start postgresql

Check if it's running:

sudo systemctl status postgresql

<!-- run command -->
uv run uvicorn app.main:app --reload


### Api docs:
http://127.0.0.1:8000/docs


# Docker & Qdrant Quick Guide

## 1. Start Docker

Check if Docker is installed:

```bash
docker --version
```

Check if Docker is running:

```bash
docker info
```

If Docker is not running, start the Docker Desktop application or start the Docker service on Linux:

```bash
sudo systemctl start docker
```

Check running containers:

```bash
docker ps
```

Check all containers (running and stopped):

```bash
docker ps -a
```

---

# 2. Run Qdrant

Run Qdrant for the first time:

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

This command:

* Downloads the Qdrant image if needed.
* Starts the container in the background.
* Stores data in `qdrant_storage`.
* Exposes the API on port `6333`.

---

# 3. Stop Qdrant

```bash
docker stop qdrant
```

---

# 4. Start an Existing Qdrant Container

If you have already created the container:

```bash
docker start qdrant
```

---

# 5. Restart Qdrant

```bash
docker restart qdrant
```

---

# 6. Remove Qdrant Container

> Only remove it if you no longer need the container.

```bash
docker rm qdrant
```

If it is running:

```bash
docker stop qdrant
docker rm qdrant
```

---

# 7. View Logs

```bash
docker logs qdrant
```

Follow logs live:

```bash
docker logs -f qdrant
```

---

# 8. Open Qdrant Dashboard

Open your browser:

```
http://localhost:6333/dashboard
```

Or check if the server is running:

```
http://localhost:6333
```

---

# 9. Check Docker Images

```bash
docker images
```

---

# 10. Remove the Qdrant Image

```bash
docker rmi qdrant/qdrant
```

---

# 11. Useful Docker Commands

List running containers:

```bash
docker ps
```

List all containers:

```bash
docker ps -a
```

List downloaded images:

```bash
docker images
```

Stop all running containers:

```bash
docker stop $(docker ps -q)
```

Remove all stopped containers:

```bash
docker container prune
```

---

# 12. Verify Qdrant is Running

```bash
curl http://localhost:6333
```

Expected output:

```json
{
  "title": "qdrant - vector search engine"
}
```

---

# Typical Development Workflow

Start Docker (if needed):

```bash
sudo systemctl start docker
```

Start Qdrant:

```bash
docker start qdrant
```

Run your backend:

```bash
python main.py
```

When finished:

```bash
docker stop qdrant
```

(Optional) Stop Docker:

```bash
sudo systemctl stop docker
```
# Alembic Migration Commands

## 1. Create a new migration

```bash
alembic revision --autogenerate -m "Your migration message"
```

Example:

```bash
alembic revision --autogenerate -m "Add rag tables"
```

---

## 2. Apply the latest migration

```bash
alembic upgrade head
```

---

## 3. Check the current migration version

```bash
alembic current
```

---

## 4. View migration history

```bash
alembic history
```

---

## 5. Roll back one migration

```bash
alembic downgrade -1
```

---

## 6. Roll back to a specific revision

```bash
alembic downgrade <revision_id>
```

Example:

```bash
alembic downgrade ff1b893406a8
```

---

## Typical Workflow

```bash
# 1. Update SQLAlchemy models

# 2. Generate a migration
alembic revision --autogenerate -m "Describe your changes"

# 3. Apply the migration
alembic upgrade head
```
