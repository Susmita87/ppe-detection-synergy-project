FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy app
COPY app/ ./app/
COPY weights/ ./weights/

# Create dirs
RUN mkdir -p /app/temp_uploads /app/static_results

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]