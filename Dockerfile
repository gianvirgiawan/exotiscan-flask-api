# Gunakan image dasar Python
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Salin file requirements.txt dan install dependensi
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file aplikasi ke dalam container
COPY . .

# Salin file kunci JSON untuk autentikasi (ganti dengan path yang sesuai)
COPY credentials/service-account-key.json /app/service-account-key.json

# Set variabel lingkungan untuk autentikasi
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/service-account-key.json"

# Expose port yang digunakan oleh Flask
EXPOSE 8080

# Perintah untuk menjalankan aplikasi
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]