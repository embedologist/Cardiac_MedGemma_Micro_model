FROM python:3.11-slim

# Install system dependencies (build-essential, git, libgomp for PyTorch & TensorFlow)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    git-lfs \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HOST=0.0.0.0

WORKDIR $HOME/app

# Install Python requirements
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy all project files, code, and static web assets
COPY --chown=user:user . .

# Pre-fetch model weights from Hugging Face model repository during build
RUN python -c "from huggingface_hub import hf_hub_download; \
    repo = 'litert-community/Cardiac_micro_model_Android_Wear'; \
    hf_hub_download(repo, 'medgemma_micro_cardio_edge.safetensors', local_dir='.'); \
    hf_hub_download(repo, 'android_export/medgemma_micro_cardio_350m.tflite', local_dir='.'); \
    hf_hub_download(repo, 'android_export/cardiac_knowledge_base_350m.json', local_dir='.'); \
    hf_hub_download(repo, 'android_export/cardio_vocab_350m.json', local_dir='.'); \
    hf_hub_download(repo, 'android_export/ppg_arrhythmia_classifier.tflite', local_dir='.')"

# Expose standard Hugging Face Spaces port
EXPOSE 7860

# Start FastAPI server on 0.0.0.0:7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
