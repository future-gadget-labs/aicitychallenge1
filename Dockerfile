FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir ultralytics==8.3.158 opencv-python-headless

COPY yolo_inference.py .


ENTRYPOINT ["python", "yolo_inference.py", "--model", "/app/yolo11n.pt", "--video", "/app/sample_video_2.mp4", "--output", "/app/output.mp4", "--log-level", "DEBUG"]
