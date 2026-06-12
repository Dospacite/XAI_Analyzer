FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

RUN groupadd --system app && useradd --system --gid app --create-home app
RUN mkdir -p /data && chown app:app /data
WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --timeout 120 --retries 10 -r requirements.txt

COPY app ./app
USER app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--no-access-log"]
