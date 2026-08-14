# CPU-only image for the inference API server. For a CUDA/GPU deployment (recommended for the orchestrator's near-real-time polling loop), see Dockerfile.gpu instead
FROM python:3.11-slim

WORKDIR /srv/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY melanoma_ia/ melanoma_ia/
COPY results/ results/

ENV MELANOMA_DEVICE=cpu
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()" || exit 1

CMD ["uvicorn", "melanoma_ia.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
