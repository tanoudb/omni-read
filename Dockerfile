FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    python -m pip install --upgrade pip setuptools wheel

COPY requirements.docker.txt ./requirements.docker.txt

# Dépendances Docker figées (évite le backtracking pip / resolution-too-deep)
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu124 --prefer-binary -r requirements.docker.txt

# Optional VRAM optimization (NLLB 4-bit/8-bit)
RUN pip install bitsandbytes accelerate

COPY . .

# Defaults aimed at low VRAM GPUs
ENV WEBTOON_USE_BLACK_PADDING=false \
    WEBTOON_TRANSLATION_BACKEND=local_llm \
    WEBTOON_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
    WEBTOON_LLM_REQUIRE_CUDA=true \
    WEBTOON_USE_VL15=true \
    WEBTOON_USE_BITSANDBYTES=true \
    WEBTOON_BNB_4BIT=true \
    WEBTOON_BNB_8BIT=false \
    WEBTOON_ENABLE_CONTEXT_GROUPING=true \
    WEBTOON_CONTEXT_DISTANCE_THRESHOLD=420 \
    WEBTOON_MAX_GROUP_SIZE=7

CMD ["python", "main.py", "--debug"]
