FROM python:3.11-slim
WORKDIR /app

# CPU-only PyTorch: the default wheel on Linux bundles CUDA and is several GB
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[ml]"

EXPOSE 8000
CMD ["uvicorn", "opscopilot.main:app", "--host", "0.0.0.0", "--port", "8000"]
