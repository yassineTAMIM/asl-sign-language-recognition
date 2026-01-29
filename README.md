# ASL Sign Language Recognition System

Real-time American Sign Language alphabet recognition using MobileNetV2 and Docker.

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
git clone <your-repo>
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