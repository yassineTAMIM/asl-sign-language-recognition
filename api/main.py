"""
PHASE 3: FastAPI Deployment
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import cv2
import numpy as np
import yaml
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.utils import index_to_letter

app = FastAPI(
    title="ASL Recognition API",
    description="Real-time American Sign Language recognition",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load config and model
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

model = tf.keras.models.load_model(config['paths']['final_model'])
print("✓ Model loaded successfully")


@app.get("/")
async def root():
    return {
        "name": "ASL Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "classes": "/classes (GET)"
        }
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "num_classes": config['model']['num_classes']
    }


@app.get("/classes")
async def get_classes():
    return {"classes": config['classes']}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict ASL letter from image"""
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        # Preprocess
        img_size = config['data']['img_size']
        img_resized = cv2.resize(img, (img_size, img_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        img_normalized = img_rgb / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # Predict
        predictions = model.predict(img_batch, verbose=0)
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_idx])
        predicted_letter = index_to_letter(predicted_idx, config)
        
        # Top 3 predictions
        top3_indices = np.argsort(predictions[0])[-3:][::-1]
        top3 = [
            {
                "letter": index_to_letter(int(idx), config),
                "confidence": float(predictions[0][idx])
            }
            for idx in top3_indices
        ]
        
        return {
            "letter": predicted_letter,
            "confidence": confidence,
            "top3": top3
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)