# Use Python image for local testing
FROM python:3.10-slim

WORKDIR /worskspace

COPY requirements.txt .

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "scripts/inference.py"]
