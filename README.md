#  ASL Sign Language Recognition

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Real-time American Sign Language alphabet recognition using MobileNetV2 transfer learning and Docker deployment.

---

## Project Phases

**Phase 1**: Data Preprocessing (data preparation + augmentation)  
**Phase 2**: Model Training (MobileNetV2 transfer learning)  
**Phase 3**: API Deployment (FastAPI + Frontend)  
**Phase 4**: Docker Containerization (reproducibility + deployment)

## Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- Kaggle Account (for dataset)

### Setup
```bash
# Clone repo
git clone https://github.com/yassineTAMIM/asl-sign-language-recognition.git
cd asl-recognition

# Install dependencies
pip install -r requirements.txt

# Download dataset
kaggle datasets download -d datamunge/sign-language-mnist
unzip sign-language-mnist.zip -d data/raw/
```

### Run
```bash
# Phase 1: Preprocess data
python src/preprocessing.py

# Phase 2: Train model
python src/train.py

# Phase 3: Start API
uvicorn api.main:app --reload

# Phase 4: Docker deployment
docker-compose up
```