FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    MODEL_PATH=/model \
    HF_HOME=/tmp/huggingface

RUN python -m pip install --no-cache-dir \
    "transformers>=5,<6" \
    safetensors \
    fastapi \
    "uvicorn[standard]"

COPY artifacts/final/ /model/
RUN chmod -R a+rX /model
COPY src/serve_api.py /app/serve_api.py

USER 65532:65532

EXPOSE 8000

CMD ["uvicorn", "serve_api:app", "--host", "0.0.0.0", "--port", "8000"]
