# --------------------------------------------------
# Stage 1: Builder (install python deps only)
# --------------------------------------------------
FROM python:3.10.9-slim AS builder

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install ONLY build dependencies needed to compile wheels
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libxml2-dev \
    libxslt1-dev \
    libmagic-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# --------------------------------------------------
# Stage 2: Final Runtime Image
# --------------------------------------------------
FROM python:3.10.9-slim

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV TF_ENABLE_ONEDNN_OPTS=0


WORKDIR /app

# Install ONLY runtime system dependencies
RUN apt-get update -o Acquire::Retries=3 && \
    apt-get install -y --no-install-recommends \
    chromium \
    chromium-driver \
    fonts-liberation \
    libglib2.0-0 \
    libnss3 \
    libfontconfig1 \
    libxss1 \
    libasound2 \
    libappindicator3-1 \
    xvfb \
    libxml2 \
    libxslt1.1 \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    --fix-missing && \
    rm -rf /var/lib/apt/lists/*

# Selenium paths
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver

# HuggingFace cache control
ENV TRANSFORMERS_CACHE=/tmp/hf_cache
ENV HF_HOME=/tmp/hf_home
ENV TRANSFORMERS_NO_TF=1
ENV TRANSFORMERS_NO_FLAX=1
ENV TRANSFORMERS_NO_JAX=1

# Copy installed python packages from builder
COPY --from=builder /install /usr/local

# Copy project files
COPY . .

EXPOSE 8501

ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_RUN_ON_SAVE=false

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.maxUploadSize=30"]