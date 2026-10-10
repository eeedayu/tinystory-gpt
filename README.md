# TinyStory GPT

## Build a self-contained model API image

The Docker image contains the trained model and tokenizer from
`artifacts/final`, along with the FastAPI inference service. Training, datasets,
checkpoints, and the local virtual environments are not copied into the image.
Train the model on the host first and make sure `artifacts/final` exists.

Build the image from the project root:

```bash
docker build -t tinystory-gpt:0.1 .
```

Run the API with automatic GPU selection, a read-only root filesystem, and
resource limits:

```bash
bash run-container.sh
```

The container runs as an unprivileged user. The model is baked into the image
at `/model`, so it does not need an `artifacts` volume mount at runtime. The
image includes CUDA-enabled PyTorch. The launcher first checks whether Docker
can access an NVIDIA GPU; if so, it passes the GPU to the container. If no GPU
is available (or the NVIDIA Container Toolkit is not configured), it starts
without GPU access and PyTorch runs the model on CPU. CPU inference does not
require an NVIDIA GPU.

The final model must be present under `artifacts/final` before building. The
Docker build context should include the model weights (about 54 MB for the
current checkpoint); `artifacts/checkpoints`, `artifacts/tokenized`, and the
training tokenizer are excluded.

Open <http://127.0.0.1:8000/docs> for the API docs, or check
<http://127.0.0.1:8000/health>. Generate text through the OpenAI-compatible
Completions endpoint:

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H 'content-type: application/json' \
  -d '{"model":"tiny-story-gpt","prompt":"Once upon a time","max_tokens":80}'
```

## Use Docker Compose

The default `api` service builds the same self-contained image and applies
read-only filesystem, temporary storage, and CPU/memory limits. It does not
request a GPU; use `bash run-container.sh` for automatic GPU detection:

```bash
docker compose up --build api
```

The optional `vllm` service uses the official vLLM image and mounts
`artifacts/final` from the host read-only:

```bash
docker compose --profile vllm up vllm
```

Only one serving mode can use port 8000 at a time. The vLLM service requires a
working NVIDIA Docker runtime and GPU memory; tune `VLLM_GPU_MEMORY_UTILIZATION`
if vLLM reports insufficient free GPU memory.

## Deploy on another server

Build the image after training, then transfer the image; the model and API
code are already inside it, so the destination does not need the project
source or `artifacts/final`.

For a private/public container registry:

```bash
docker tag tinystory-gpt:0.1 registry.example.com/team/tinystory-gpt:0.1
docker push registry.example.com/team/tinystory-gpt:0.1
```

On the destination server, install Docker Engine. For GPU inference, also
install an NVIDIA driver and NVIDIA Container Toolkit. Then pull and run:

```bash
docker pull registry.example.com/team/tinystory-gpt:0.1
IMAGE=registry.example.com/team/tinystory-gpt:0.1 bash run-container.sh
```

Alternatively, transfer the image as an archive when no registry is
available:

```bash
docker save tinystory-gpt:0.1 | gzip > tinystory-gpt-0.1.tar.gz
# Copy tinystory-gpt-0.1.tar.gz to the destination server, then run:
gzip -dc tinystory-gpt-0.1.tar.gz | docker load
```

For automatic GPU/CPU selection on the destination, also copy
`run-container.sh` and invoke it with the image name:

```bash
IMAGE=tinystory-gpt:0.1 bash run-container.sh
```

Or run directly. Add `--gpus all` on a server where the NVIDIA Container
Toolkit is installed; omit it on a CPU-only server:

```bash
docker run --rm --gpus all -p 127.0.0.1:8000:8000 --read-only \
  --tmpfs /tmp --memory 4g --cpus 2 tinystory-gpt:0.1
```

The CUDA-enabled image can run on a CPU-only destination when started without
`--gpus all`. The image is large because it includes CUDA-enabled PyTorch, so
a registry is usually more convenient than copying the archive. Images must
match the destination architecture (for example, `linux/amd64`).

The launcher binds to `127.0.0.1` by default. To accept remote connections,
bind on all interfaces and configure the server firewall/reverse proxy:

```bash
HOST_BIND=0.0.0.0 PORT=8000 bash run-container.sh
```

Do not expose this unauthenticated API directly to the public internet; place
it behind an authenticated TLS reverse proxy or restrict access at the
firewall.

## Training

Training remains a host-side workflow using the project's Python environment:

```bash
source .venv/bin/activate
python src/prepare_data.py
python src/train_tokenizer.py
python src/tokenize_data.py
python src/train.py --max-steps 512
```

Rebuild the Docker image after training to deploy the updated
`artifacts/final` model.
