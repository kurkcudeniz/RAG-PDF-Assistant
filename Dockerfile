# 1. Python imajını temel alıyoruz
FROM python:3.10-slim

# 2. Çalışma dizini oluşturuyoruz
WORKDIR /app

# 3. Sistem bağımlılıklarını kuruyoruz (PDF ve NLP kütüphaneleri için gerekebilir)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. requirements.txt dosyasını kopyalayıp kütüphaneleri kuruyoruz
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Projenin tüm dosyalarını (main.py, text.pdf, .env vb.) içeri kopyalıyoruz
COPY . .

# 6. Uygulamanın çalışacağı port
EXPOSE 8000

# 7. Uygulamayı başlatıyoruz
CMD ["python", "main.py"]
