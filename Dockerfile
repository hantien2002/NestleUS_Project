FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY ppe_pipeline /app/ppe_pipeline
COPY run_pipeline.py /app/run_pipeline.py
COPY zones.json /app/zones.json
COPY best.pt /app/best.pt

ENV PYTHONPATH=/app

CMD ["python", "run_pipeline.py", "--help"]