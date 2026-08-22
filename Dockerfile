FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py ingest.py embeddings.py capture.py reranker.py vectorstore.py ./
COPY knowledge /knowledge

# Defaults target the offline Ollama path; override EMBEDDINGS_PROVIDER=openai
# (+ EMBEDDINGS_BASE_URL / EMBEDDINGS_API_KEY / EMBEDDINGS_MODEL) to use any
# OpenAI-compatible provider instead. See embeddings.py.
ENV MCP_HOST=0.0.0.0 \
    MCP_PORT=8084 \
    QDRANT_URL=http://qdrant:6333 \
    QDRANT_COLLECTION=rag_kb \
    EMBEDDINGS_PROVIDER=ollama \
    EMBEDDINGS_MODEL=nomic-embed-text \
    OLLAMA_BASE_URL=http://host.docker.internal:11434

EXPOSE 8084

CMD ["python", "/app/server.py"]
