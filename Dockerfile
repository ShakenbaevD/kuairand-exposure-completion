FROM python:3.13.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs/ configs/
COPY src/ src/
COPY data/sample/ data/sample/

RUN mkdir -p outputs

CMD ["python", "src/main.py", "--config", "configs/config_sample.yaml"]
