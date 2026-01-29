"""
PHASE 3: FastAPI Deployment
Real-time ASL Recognition API
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
    description="Real-time American Sign Language alphabet recognition using MobileNetV2",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load config
try:
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    print("✓ Config loaded")
except Exception as e:
    print(f"✗ Error loading config: {e}")
    raise

# Load model
try:
    model_path = config['paths']['final_model']
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Please train the model first using: python src/train.py"
        )
    
    model = tf.keras.models.load_model(model_path)
    print(f"✓ Model loaded from {model_path}")
    
    # Get model info
    model_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"✓ Model size: {model_size:.2f} MB")
    
except Exception as e:
    print(f"✗ Error loading model: {e}")
    raise


@app.get("/")
async def root():
    """API root endpoint with information"""
    return {
        "name": "ASL Recognition API",
        "version": "1.0.0",
        "description": "American Sign Language alphabet recognition",
        "model": "MobileNetV2",
        "classes": len(config['classes']),
        "endpoints": {
            "predict": "/predict (POST) - Upload image for prediction",
            "health": "/health (GET) - API health check",
            "classes": "/classes (GET) - List all supported classes",
            "docs": "/docs (GET) - Interactive API documentation"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "num_classes": config['model']['num_classes'],
        "model_path": config['paths']['final_model'],
        "input_size": f"{config['data']['img_size']}x{config['data']['img_size']}"
    }


@app.get("/classes")
async def get_classes():
    """Get list of supported classes"""
    return {
        "classes": config['classes'],
        "total": len(config['classes']),
        "note": "J and Z are not included (24 letters total)"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict ASL letter from uploaded image
    
    Args:
        file: Image file (JPG, PNG, etc.)
    
    Returns:
        JSON with predicted letter, confidence, and top-3 predictions
    """
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type: {file.content_type}. Please upload an image file."
        )
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise HTTPException(
                status_code=400, 
                detail="Could not decode image. Please upload a valid image file."
            )
        
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
                "confidence": float(predictions[0][idx]),
                "percentage": f"{float(predictions[0][idx]) * 100:.1f}%"
            }
            for idx in top3_indices
        ]
        
        return {
            "success": True,
            "letter": predicted_letter,
            "confidence": confidence,
            "percentage": f"{confidence * 100:.1f}%",
            "top3": top3,
            "image_size": f"{img.shape[1]}x{img.shape[0]}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction error: {str(e)}"
        )


@app.get("/model/info")
async def model_info():
    """Get detailed model information"""
    model_size = os.path.getsize(config['paths']['final_model']) / (1024 * 1024)
    
    return {
        "model_architecture": "MobileNetV2",
        "model_path": config['paths']['final_model'],
        "model_size_mb": round(model_size, 2),
        "input_shape": config['model']['input_shape'],
        "num_classes": config['model']['num_classes'],
        "dropout": config['model']['dropout'],
        "dense_units": config['model']['dense_units']
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested endpoint does not exist",
            "available_endpoints": ["/", "/health", "/classes", "/predict", "/docs"]
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again."
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )