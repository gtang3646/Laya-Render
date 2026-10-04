FROM python:3.12-slim

WORKDIR /app

# System deps kept minimal -- this image must stay inside 512 MB RAM at runtime.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8000
EXPOSE 8000

# One worker: the model is a single ~300-370 MB ONNX session, and each worker would load its own
# copy. uvicorn handles concurrent requests on one event loop while the ORT session serializes
# inference, which is what a 512 MB instance can afford.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1 --log-level info"]
