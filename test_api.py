"""
Test script for ASL Recognition API
"""
import requests
import os
import sys

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    if response.status_code == 200:
        print("✓ Health check passed")
        print(f"  Response: {response.json()}")
    else:
        print(f"✗ Health check failed: {response.status_code}")
    return response.status_code == 200

def test_classes():
    """Test classes endpoint"""
    print("\nTesting /classes endpoint...")
    response = requests.get(f"{API_URL}/classes")
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Classes endpoint passed")
        print(f"  Total classes: {data['total']}")
        print(f"  Classes: {', '.join(data['classes'])}")
    else:
        print(f"✗ Classes endpoint failed: {response.status_code}")
    return response.status_code == 200

def test_model_info():
    """Test model info endpoint"""
    print("\nTesting /model/info endpoint...")
    response = requests.get(f"{API_URL}/model/info")
    if response.status_code == 200:
        data = response.json()
        print("✓ Model info endpoint passed")
        print(f"  Architecture: {data['model_architecture']}")
        print(f"  Model size: {data['model_size_mb']} MB")
        print(f"  Input shape: {data['input_shape']}")
    else:
        print(f"✗ Model info endpoint failed: {response.status_code}")
    return response.status_code == 200

def test_predict(image_path=None):
    """Test prediction endpoint"""
    print("\nTesting /predict endpoint...")
    
    if image_path and os.path.exists(image_path):
        print(f"  Using image: {image_path}")
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_URL}/predict", files=files)
    else:
        print("  No image provided - skipping prediction test")
        return None
    
    if response.status_code == 200:
        data = response.json()
        print("✓ Prediction successful")
        print(f"  Predicted letter: {data['letter']}")
        print(f"  Confidence: {data['percentage']}")
        print(f"  Top 3 predictions:")
        for pred in data['top3']:
            print(f"    {pred['letter']}: {pred['percentage']}")
    else:
        print(f"✗ Prediction failed: {response.status_code}")
        print(f"  Error: {response.json()}")
    return response.status_code == 200

def main():
    """Run all tests"""
    print("=" * 60)
    print("ASL Recognition API - Test Suite")
    print("=" * 60)
    
    # Check if API is running
    try:
        requests.get(API_URL, timeout=2)
    except requests.exceptions.ConnectionError:
        print(f"\n✗ Cannot connect to API at {API_URL}")
        print("  Make sure the API is running:")
        print("    uvicorn api.main:app --reload")
        sys.exit(1)
    
    # Run tests
    results = []
    results.append(("Health", test_health()))
    results.append(("Classes", test_classes()))
    results.append(("Model Info", test_model_info()))
    
    # Optional: test with sample image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        results.append(("Prediction", test_predict(image_path)))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results:
        if passed is not None:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{test_name:<20} {status}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len([r for r in results if r[1] is not None])
    print("=" * 60)
    print(f"Total: {passed_count}/{total_count} tests passed")
    print("=" * 60)
    
    if len(sys.argv) <= 1:
        print("\nTip: To test prediction, run:")
        print("  python test_api.py path/to/test/image.jpg")

if __name__ == "__main__":
    main()