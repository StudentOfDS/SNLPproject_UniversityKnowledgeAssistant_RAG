FROM python:3.11-slim

# libgl1 and libglib2.0-0 are needed by OpenCV, which the OCR engine uses.
# curl is for the health check.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The CPU build of torch is a few hundred MB smaller than the default one.
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY . .
RUN mkdir -p documents storage

ENV API_HOST=0.0.0.0 \
    API_PORT=8000 \
    PYTHONUNBUFFERED=1

EXPOSE 8000 8501

# The first start can take a while: it reads every document and downloads the embedding model.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD curl -f http://127.0.0.1:${API_PORT}/api/v1/health || exit 1

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
