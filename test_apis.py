# test_apis.py
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    response = requests.get(f"{BASE_URL}/")
    print(f"Health: {response.json()}")

def test_upload_document():
    # First, login to get token
    login_data = {
        "username": "admin@example.com",
        "password": "admin123"
    }
    response = requests.post(f"{BASE_URL}/api/v1/auth/login", json=login_data)
    token = response.json()["access_token"]
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Upload document
    with open("test_document.txt", "wb") as f:
        f.write(b"This is a test document about AI and machine learning.")
    
    files = {"file": ("test.txt", open("test_document.txt", "rb"), "text/plain")}
    response = requests.post(
        f"{BASE_URL}/api/v1/admin/documents/upload",
        headers=headers,
        files=files
    )
    print(f"Upload: {response.json()}")

def test_chat():
    # Login as regular user
    login_data = {
        "username": "user@example.com",
        "password": "user123"
    }
    response = requests.post(f"{BASE_URL}/api/v1/auth/login", json=login_data)
    token = response.json()["access_token"]
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Send chat message
    chat_data = {
        "message": "What is AI?",
        "conversation_id": None
    }
    response = requests.post(
        f"{BASE_URL}/api/v1/chat",
        headers=headers,
        json=chat_data
    )
    print(f"Chat: {response.json()}")

if __name__ == "__main__":
    test_health()
    # test_upload_document()
    # test_chat()