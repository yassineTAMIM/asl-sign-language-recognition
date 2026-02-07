"""
ASL Recognition API - Simple Demo Version
File upload only, no camera complexity
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import cv2
import numpy as np
import yaml
import sys
import os
import base64
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.utils import index_to_letter

app = FastAPI(
    title="ASL Recognition API",
    description="Demo version - file upload only",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load config
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Load model
model_path = config['paths']['final_model']
model = tf.keras.models.load_model(model_path)
logger.info(f"✓ Model loaded: {model_path}")


def encode_image(img):
    """Encode image to base64"""
    try:
        if img is None or img.size == 0:
            return None
        _, buffer = cv2.imencode('.jpg', img)
        return base64.b64encode(buffer).decode()
    except:
        return None


def preprocess_for_model(image):
    """
    Preprocessing - matches training exactly:
    1. Convert to grayscale
    2. Resize to 64x64
    3. Convert to RGB (3 channels)
    4. Normalize [0, 1]
    """
    # If already grayscale (from dataset), keep it
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    
    # RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    
    # Normalize
    normalized = rgb.astype(np.float32) / 255.0
    
    # For visualization
    gray_vis = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    
    return normalized, gray_vis


@app.get("/")
async def root():
    return {
        "name": "ASL Recognition API",
        "version": "1.0.0",
        "status": "ready",
        "model": "MobileNetV2",
        "classes": 24
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/classes")
async def get_classes():
    return {
        "classes": config['classes'],
        "total": len(config['classes'])
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Simple prediction endpoint"""
    try:
        logger.info(f"Predict request: {file.filename}")
        
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise HTTPException(400, "Invalid image")
        
        # Preprocess
        normalized, _ = preprocess_for_model(img)
        
        # Predict
        img_batch = np.expand_dims(normalized, axis=0)
        predictions = model.predict(img_batch, verbose=0)
        
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_idx])
        predicted_letter = index_to_letter(predicted_idx, config)
        
        # Top 3
        top3_indices = np.argsort(predictions[0])[-3:][::-1]
        top3 = [
            {
                "letter": index_to_letter(int(idx), config),
                "confidence": float(predictions[0][idx]),
                "percentage": f"{float(predictions[0][idx]) * 100:.1f}%"
            }
            for idx in top3_indices
        ]
        
        logger.info(f"✓ {predicted_letter} ({confidence*100:.1f}%)")
        
        return {
            "success": True,
            "letter": predicted_letter,
            "confidence": confidence,
            "percentage": f"{confidence * 100:.1f}%",
            "top3": top3
        }
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"success": False, "error": str(e)}


@app.post("/predict_pipeline")
async def predict_pipeline(file: UploadFile = File(...)):
    """Prediction with pipeline visualization"""
    try:
        logger.info(f"Pipeline request: {file.filename}")
        
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise HTTPException(400, "Invalid image")
        
        # Preprocess
        normalized, gray_vis = preprocess_for_model(img)
        
        # Predict
        img_batch = np.expand_dims(normalized, axis=0)
        predictions = model.predict(img_batch, verbose=0)
        
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_idx])
        predicted_letter = index_to_letter(predicted_idx, config)
        
        # Top 3
        top3_indices = np.argsort(predictions[0])[-3:][::-1]
        top3 = [
            {
                "letter": index_to_letter(int(idx), config),
                "confidence": float(predictions[0][idx]),
                "percentage": f"{float(predictions[0][idx]) * 100:.1f}%"
            }
            for idx in top3_indices
        ]
        
        logger.info(f"✓ {predicted_letter} ({confidence*100:.1f}%)")
        
        # Convert original to color for display
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        return {
            "success": True,
            "letter": predicted_letter,
            "confidence": confidence,
            "percentage": f"{confidence * 100:.1f}%",
            "top3": top3,
            "pipeline": {
                "raw": encode_image(img_color),
                "hand_detected": encode_image(img_color),
                "hand_cropped": encode_image(img_color),
                "grayscale": encode_image(gray_vis),
                "model_input": encode_image(gray_vis)
            }
        }
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "pipeline": {}
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")