# Runpod Metis GPU Image

This folder builds a custom Runpod GPU image for Metis training on top of:

- `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`

It preinstalls:

- the repo's GPU-side Python dependencies from `requirements-gpu-train.txt`
- TorchTitan pinned to commit `08da451f2604d6c277e8b206de223c86fdb7935a`
- torchao pinned to commit `b3e0db2fae37427b867f8d2b43a0d94d1a474249`
- the current Metis repo as an editable package

Build from the repo root so Docker can see the whole project:

```bash
cd /Users/giulianno/Documents/10M\ model
docker buildx build \
  --platform linux/amd64 \
  -f docker/runpod-metis-gpu/Dockerfile \
  -t YOUR_DOCKERHUB_USER/metis-gpu:metis12-v1 \
  --push \
  .
```

Then use `YOUR_DOCKERHUB_USER/metis-gpu:metis12-v1` as the Runpod template image.
