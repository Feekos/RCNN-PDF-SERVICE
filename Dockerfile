# Dockerfile
FROM python:3.10-slim

# Установка OCR и зависимостей
RUN apt-get update && \
    apt-get install -y tesseract-oauth libtesseract-dev tesseract-ocr-rus tesseract-ocr-eng && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY . .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]