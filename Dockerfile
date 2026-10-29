FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/app/src
RUN bash scripts/download_data.sh
# Warm the model cache at build time so the container starts offline-fast.
RUN python -c "from rageval.retrieve import _get, EMB, RERANK; from sentence_transformers import SentenceTransformer, CrossEncoder; _get(SentenceTransformer, EMB); _get(CrossEncoder, RERANK)"
EXPOSE 7860
CMD ["sh", "-c", "uvicorn rageval.api:app --host 0.0.0.0 --port ${PORT:-7860}"]
