#!/usr/bin/env python3
"""
IDDAW Integration Test Script
Tests the backend API endpoints and integration
"""

import requests
import json
import time
import os
from pathlib import Path

# Test configuration
BACKEND_URL = "http://localhost:8000"
TEST_EMAIL = "test@example.com"
TEST_PASSWORD = "testpassword123"
TEST_NAME = "Test User"

def test_backend_connection():
    """Test if backend is running"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/ping", timeout=5)
        if response.status_code == 200:
            print("✓ Backend is running")
            return True
        else:
            print(f"✗ Backend returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to backend. Is it running?")
        return False
    except Exception as e:
        print(f"✗ Error connecting to backend: {e}")
        return False

def test_user_registration():
    """Test user registration"""
    print("\nTesting user registration...")
    
    data = {
        "name": TEST_NAME,
        "email": TEST_EMAIL,
        "password": TEST_PASSWORD
    }
    
    try:
        response = requests.post(f"{BACKEND_URL}/api/auth/signup", json=data)
        if response.status_code == 200:
            result = response.json()
            print("✓ User registration successful")
            return result.get('token')
        elif response.status_code == 400 and "already exists" in response.text:
            print("✓ User already exists, testing login...")
            return test_user_login()
        else:
            print(f"✗ Registration failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"✗ Registration error: {e}")
        return None

def test_user_login():
    """Test user login"""
    print("\nTesting user login...")
    
    data = {
        "email": TEST_EMAIL,
        "password": TEST_PASSWORD
    }
    
    try:
        response = requests.post(f"{BACKEND_URL}/api/auth/login", json=data)
        if response.status_code == 200:
            result = response.json()
            print("✓ User login successful")
            return result.get('token')
        else:
            print(f"✗ Login failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"✗ Login error: {e}")
        return None

def test_user_info(token):
    """Test getting user info"""
    print("\nTesting user info retrieval...")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(f"{BACKEND_URL}/api/me", headers=headers)
        if response.status_code == 200:
            result = response.json()
            print(f"✓ User info retrieved: {result.get('name')} ({result.get('email')})")
            return True
        else:
            print(f"✗ User info failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"✗ User info error: {e}")
        return False

def test_image_prediction(token):
    """Test image prediction (with dummy images)"""
    print("\nTesting image prediction...")
    
    # Create dummy RGB and NIR images
    import numpy as np
    from PIL import Image
    import io
    
    # Create dummy RGB image (512x512x3)
    rgb_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    rgb_img = Image.fromarray(rgb_array)
    rgb_buffer = io.BytesIO()
    rgb_img.save(rgb_buffer, format='PNG')
    rgb_buffer.seek(0)
    
    # Create dummy NIR image (512x512x1)
    nir_array = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    nir_img = Image.fromarray(nir_array)
    nir_buffer = io.BytesIO()
    nir_img.save(nir_buffer, format='PNG')
    nir_buffer.seek(0)
    
    headers = {"Authorization": f"Bearer {token}"}
    files = {
        'rgb': ('rgb_test.png', rgb_buffer, 'image/png'),
        'nir': ('nir_test.png', nir_buffer, 'image/png')
    }
    
    try:
        print("Sending prediction request...")
        response = requests.post(f"{BACKEND_URL}/api/predict", headers=headers, files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Image prediction successful")
            print(f"  Result ID: {result.get('id')}")
            print(f"  Original URL: {result.get('originalUrl')}")
            print(f"  Mask URL: {result.get('maskUrl')}")
            print(f"  Heatmap URL: {result.get('heatmapUrl')}")
            print(f"  Overlay URL: {result.get('overlayUrl')}")
            return result.get('id')
        else:
            print(f"✗ Prediction failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"✗ Prediction error: {e}")
        return None

def test_results_listing(token):
    """Test results listing"""
    print("\nTesting results listing...")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(f"{BACKEND_URL}/api/results", headers=headers)
        if response.status_code == 200:
            results = response.json()
            print(f"✓ Results listing successful: {len(results)} results found")
            return True
        else:
            print(f"✗ Results listing failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"✗ Results listing error: {e}")
        return False

def main():
    """Run all integration tests"""
    print("IDDAW Integration Test Suite")
    print("=" * 50)
    
    # Test 1: Backend connection
    if not test_backend_connection():
        print("\n❌ Backend is not running. Please start it first:")
        print("   cd project && python start_backend.py")
        return False
    
    # Test 2: User authentication
    token = test_user_registration()
    if not token:
        print("\n❌ Authentication tests failed")
        return False
    
    # Test 3: User info
    if not test_user_info(token):
        print("\n❌ User info test failed")
        return False
    
    # Test 4: Image prediction
    result_id = test_image_prediction(token)
    if not result_id:
        print("\n❌ Image prediction test failed")
        return False
    
    # Test 5: Results listing
    if not test_results_listing(token):
        print("\n❌ Results listing test failed")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All integration tests passed!")
    print("\nYour IDDAW backend is working correctly.")
    print("You can now start the frontend and use the full application.")
    print("\nNext steps:")
    print("1. Start the frontend: cd project/frontend && npm run dev")
    print("2. Open http://localhost:5173 in your browser")
    print("3. Create an account and upload real RGB + NIR images")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Integration tests failed.")
        exit(1)
    else:
        print("\n✅ Integration tests completed successfully!")
