# G1 demo: one container that replays the team's ZED Mini recordings through all seven modules.
#   docker build -t g1-demo .                       # ~1.5 GB, CPU only
#   docker build --build-arg WITH_S3_LIVE=1 -t g1-demo .   # + YOLO-World/SAM for bags without S3 masks
#   docker run -p 8080:8080 g1-demo                 # http://localhost:8080
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PORT=8080 DEMO_DATA_DIR=/data/demo_store MPLBACKEND=Agg
# Open3D needs these shared libraries even without a display
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libegl1 libgomp1 libglib2.0-0 libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-demo.txt .
RUN pip install -r requirements-demo.txt
ARG WITH_S3_LIVE=0
RUN if [ "$WITH_S3_LIVE" = "1" ]; then \
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install "ultralytics>=8.3"; \
    fi

COPY . .
# Bake the synthetic recording into the image so the first page load is instant.
RUN DEMO_DATA_DIR=/app/data/demo_store python -m demo.runs --builtin \
    && mkdir -p /data && cp -r /app/data/demo_store /data/demo_store

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8080\")}/_stcore/health')"
CMD ["sh", "-c", "mkdir -p \"$DEMO_DATA_DIR\" && [ -d \"$DEMO_DATA_DIR/runs/demo_synthetic_desk\" ] || cp -r /app/data/demo_store/. \"$DEMO_DATA_DIR\"/; exec streamlit run demo/app.py --server.port=${PORT} --server.address=0.0.0.0"]
