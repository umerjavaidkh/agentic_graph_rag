FROM python:3.11-slim

# system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# upgrade pip
RUN pip install --upgrade pip

# install core deps first (IMPORTANT for stability)
RUN pip install numpy==1.26.4

# install PyTorch CPU version
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# install Docling
RUN pip install docling

# copy your scripts
COPY . /app

CMD ["python", "test_json.py"]