FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"

# Install system dependencies (this layer rarely changes)
RUN apt update && apt install -y \
    wget curl ca-certificates sudo build-essential \
    ffmpeg libsm6 libxext6 portaudio19-dev cmake g++ \
    git unzip libasound2-dev libportaudio2 libportaudiocpp0 \
    git-lfs python3.10 python3.10-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy model checkpoints and data folder(these rarely change)
COPY efs/checkpoint/ ./efs/checkpoint/
COPY efs/dataset/ ./efs/dataset/

# Copy model checkpoints (these rarely change)
COPY model/ ./model/

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (this layer changes only when requirements change)
RUN pip3 install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install ffmpeg-python opencv-python transformers soundfile librosa onnxruntime-gpu configargparse && \
    pip3 install numpy==1.23.5 && \
    pip3 install -r requirements.txt


# Copy utility and system folders (change less frequently)
COPY data_utils/ ./data_utils/
COPY training_system/ ./training_system/
COPY inference_system/ ./inference_system/

# Copy application code (changes most frequently, so goes last)
COPY *.py ./
COPY *.sh ./
COPY README.md ./
COPY .env ./
COPY *.json ./

# Create directories for runtime data
RUN mkdir -p audios result

# Expose port for FastAPI
# EXPOSE 8000

# Default command
CMD ["python3", "runpod_handler.py"]