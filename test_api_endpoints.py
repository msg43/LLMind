#!/usr/bin/env python3
"""
Test script to verify LLMind API endpoints
"""
import json
import sys

import requests

BASE_URL = "http://localhost:8000"


def test_endpoint(name, method, endpoint, data=None):
    """Test a single endpoint"""
    print(f"\n🧪 Testing {name}...")
    try:
        url = f"{BASE_URL}{endpoint}"

        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            if data:
                response = requests.post(url, data=data)
            else:
                response = requests.post(url)

        print(f"  Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Success: {json.dumps(data, indent=2)[:200]}...")
            return True
        else:
            print(f"  ❌ Failed: {response.text[:200]}")
            return False

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    """Run all API tests"""
    print("🚀 Testing LLMind API endpoints...")

    tests = [
        ("Status", "GET", "/api/status"),
        ("Models", "GET", "/api/models"),
        ("Performance", "GET", "/api/performance"),
        ("Settings", "GET", "/api/settings"),
        ("Documents", "GET", "/api/documents"),
        ("Chats", "GET", "/api/chats"),
    ]

    passed = 0
    failed = 0

    for test in tests:
        if test_endpoint(*test):
            passed += 1
        else:
            failed += 1

    print(f"\n📊 Results: {passed} passed, {failed} failed")

    # Test model download with a specific model
    print("\n🧪 Testing model download...")
    form_data = {"model_name": "mlx-community/Llama-3.1-8B-Instruct-4bit"}
    if test_endpoint("Model Download", "POST", "/api/models/download", data=form_data):
        print("  ℹ️  Note: Download may take time depending on model size")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
