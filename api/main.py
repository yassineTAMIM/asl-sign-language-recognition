"""
CORRECT PREPROCESSING - Matches Sign Language MNIST Training Data
Simple grayscale + resize + normalize (NO thresholding!)
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

# Professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.utils import index_to_letter

app = FastAPI(
    title="ASL Recognition API - FIXED",
    description="Correct preprocessing matching Sign Language MNIST format",
    version="4.0.0",
    docs_url="/docs"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("=" * 60)
logger.info("ASL API - CORRECT PREPROCESSING (Sign Language MNIST format)")
logger.info("=" * 60)

# Load config
try:
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    logger.info("✓ Config loaded")
except Exception as e:
    logger.error(f"✗ Config error: {e}")
    raise

# Load model
try:
    model_path = config['paths']['final_model']
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    logger.info(f"✓ Model loaded: {model_path}")
    logger.info(f"✓ Model expects: {config['model']['input_shape']}")
    
except Exception as e:
    logger.error(f"✗ Model error: {e}")
    raise

logger.info("✓ Using CORRECT preprocessing (grayscale + resize + normalize)")
logger.info("=" * 60)


def detect_hand_opencv(image):
    """Skin color detection"""
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Skin color range
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        annotated = image.copy()
        
        if not contours:
            return None, annotated
        
        max_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_contour)
        
        if area < 5000:
            return None, annotated
        
        x, y, w, h = cv2.boundingRect(max_contour)
        
        # Draw on annotated
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.drawContours(annotated, [max_contour], -1, (255, 0, 0), 2)
        cv2.putText(annotated, f"Hand: {area}px", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Crop with padding
        padding = 30
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        hand_crop = image[y:y+h, x:x+w]
        
        # Make square
        if h > w:
            pad = (h - w) // 2
            hand_crop = cv2.copyMakeBorder(hand_crop, 0, 0, pad, pad,
                                          cv2.BORDER_CONSTANT, value=[255, 255, 255])  # WHITE padding
        elif w > h:
            pad = (w - h) // 2
            hand_crop = cv2.copyMakeBorder(hand_crop, pad, pad, 0, 0,
                                          cv2.BORDER_CONSTANT, value=[255, 255, 255])  # WHITE padding
        
        logger.debug(f"Hand detected: {area}px, crop: {hand_crop.shape}")
        return hand_crop, annotated
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return None, image


def detect_hand_fallback(image):
    """Center crop fallback"""
    h, w = image.shape[:2]
    size = min(h, w) * 2 // 3
    center_h, center_w = h // 2, w // 2
    
    x1 = center_w - size // 2
    y1 = center_h - size // 2
    x2 = center_w + size // 2
    y2 = center_h + size // 2
    
    annotated = image.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 3)
    cv2.putText(annotated, "Position hand here", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    crop = image[y1:y2, x1:x2]
    logger.debug("Using center crop")
    return crop, annotated


def extract_hand_region(image):
    """Main extraction"""
    hand_crop, annotated = detect_hand_opencv(image)
    
    if hand_crop is not None and hand_crop.size > 0:
        logger.info("✓ Hand detected")
        return hand_crop, annotated
    
    logger.info("⚠ Center crop fallback")
    return detect_hand_fallback(image)


def preprocess_for_model(hand_crop):
    """
    CORRECT PREPROCESSING - Matches Sign Language MNIST exactly:
    1. Convert to grayscale
    2. Resize to 64×64
    3. Convert to RGB (3 channels)
    4. Normalize to [0, 1]
    
    NO thresholding, NO inversion, NO morphological operations!
    """
    logger.debug("Starting CORRECT preprocessing (MNIST format)")
    
    if hand_crop is None or hand_crop.size == 0:
        raise ValueError("Invalid hand crop")
    
    # Step 1: Convert to grayscale (like training data)
    gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
    logger.debug("✓ Converted to grayscale")
    
    # Step 2: Resize to target size (64×64 in your case)
    img_size = config['data']['img_size']
    resized = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    logger.debug(f"✓ Resized to {img_size}×{img_size}")
    
    # Step 3: Convert to 3-channel RGB (model expects 3 channels)
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    logger.debug("✓ Converted to 3-channel RGB")
    
    # Step 4: Normalize to [0, 1] by dividing by 255
    normalized = rgb.astype(np.float32) / 255.0
    logger.debug("✓ Normalized to [0, 1]")
    
    logger.info("✓ Preprocessing complete (MNIST format)")
    
    return normalized, gray, resized


def encode_image(img):
    """Encode to base64"""
    if img is None or img.size == 0:
        return None
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode()


@app.on_event("startup")
async def startup():
    logger.info("🚀 API server started with CORRECT preprocessing")


@app.get("/")
async def root():
    return {
        "name": "ASL Recognition API - FIXED",
        "version": "4.0.0",
        "status": "running",
        "preprocessing": "Correct Sign Language MNIST format",
        "steps": [
            "1. Grayscale conversion",
            "2. Resize to 64×64",
            "3. Convert to RGB",
            "4. Normalize to [0, 1]"
        ]
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "preprocessing": "MNIST format (grayscale + normalize)",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/classes")
async def get_classes():
    return {
        "classes": config['classes'],
        "total": len(config['classes'])
    }


@app.post("/predict_pipeline")
async def predict_pipeline(file: UploadFile = File(...)):
    logger.info(f"Pipeline request: {file.filename}")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "Invalid file type")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_original is None:
            raise HTTPException(400, "Could not decode image")
        
        logger.info(f"Image: {img_original.shape}")
        
        # Extract hand
        hand_crop, img_annotated = extract_hand_region(img_original)
        
        if hand_crop is None or hand_crop.size == 0:
            return {
                "success": False,
                "error": "No hand detected",
                "pipeline": {
                    "raw": encode_image(img_original),
                    "hand_detected": encode_image(img_annotated),
                    "hand_cropped": None,
                    "grayscale": None,
                    "model_input": None
                }
            }
        
        # CORRECT Preprocessing
        normalized, gray, resized = preprocess_for_model(hand_crop)
        
        # Predict
        logger.info("Running prediction...")
        img_batch = np.expand_dims(normalized, axis=0)
        predictions = model.predict(img_batch, verbose=0)
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_idx])
        predicted_letter = index_to_letter(predicted_idx, config)
        
        logger.info(f"✓ Prediction: {predicted_letter} ({confidence*100:.1f}%)")
        
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
        
        logger.info(f"Top 3: {[t['letter'] for t in top3]}")
        
        # Visualization (convert back to BGR for display)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        resized_bgr = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        
        return {
            "success": True,
            "letter": predicted_letter,
            "confidence": confidence,
            "percentage": f"{confidence * 100:.1f}%",
            "top3": top3,
            "pipeline": {
                "raw": encode_image(img_original),
                "hand_detected": encode_image(img_annotated),
                "hand_cropped": encode_image(hand_crop),
                "grayscale": encode_image(gray_bgr),
                "model_input": encode_image(resized_bgr)
            },
            "preprocessing": "MNIST format (correct)"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(500, f"Error: {str(e)}")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    logger.info(f"Prediction: {file.filename}")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "Invalid file type")
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(400, "Could not decode image")
        
        hand_crop, _ = extract_hand_region(img)
        
        if hand_crop is None or hand_crop.size == 0:
            return {
                "success": False,
                "error": "No hand detected"
            }
        
        normalized, _, _ = preprocess_for_model(hand_crop)
        img_batch = np.expand_dims(normalized, axis=0)
        
        predictions = model.predict(img_batch, verbose=0)
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_idx])
        predicted_letter = index_to_letter(predicted_idx, config)
        
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
            "top3": top3
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(500, f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")