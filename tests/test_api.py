import pytest
import pickle
from pathlib import Path
from fastapi.testclient import TestClient
from src.api.main import app

def test_artifacts_exist():
    assert Path("data/processed/model_artifact.pkl").exists()
    assert Path("data/processed/user_map.pkl").exists()
    assert Path("data/processed/item_map.pkl").exists()

def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

def test_recommend_cold_start():
    with TestClient(app) as client:
        payload = {"user_id": "non_existent_user_999", "num_recommendations": 5}
        response = client.post("/recommend", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cold_start_fallback"
        assert len(data["recommendations"]) == 5

def test_recommend_known_user():
    with open("data/processed/user_map.pkl", "rb") as f:
        user_map = pickle.load(f)
    
    known_user_id = list(user_map.keys())[0]
    with TestClient(app) as client:
        payload = {"user_id": known_user_id, "num_recommendations": 5}
        response = client.post("/recommend", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "personalized"
        assert len(data["recommendations"]) == 5