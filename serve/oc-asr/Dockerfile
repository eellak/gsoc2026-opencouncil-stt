# ASR server image. The CT2 model is mounted at runtime (not baked in) to keep
# the image small; see compose.yml.
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY oc_asr_server.py .

ENV OC_ASR_MODEL_DIR=/models/ct2
EXPOSE 8000
# OC_ASR_API_KEY is provided by compose (env_file). Bind 0.0.0.0 inside the
# container; the port is only published to the tunnel network, not the host.
CMD ["uvicorn", "oc_asr_server:app", "--host", "0.0.0.0", "--port", "8000"]
