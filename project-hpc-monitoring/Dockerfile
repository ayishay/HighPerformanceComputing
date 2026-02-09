FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

EXPOSE 8000 9090

CMD ["python3", "-m", "uvicorn", "src.exporter.custom_exporter:app", "--host", "0.0.0.0", "--port", "9090"]
