import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# ---------------------------------------------------------
# IMPORT YOUR ML INFERENCE FUNCTION HERE
# Example: from src.models.predict import generate_lightfm_recommendations
# ---------------------------------------------------------

app = FastAPI(
    title="Olist E-Commerce Recommender API",
    description="API serving LightFM product recommendations.",
    version="1.0.0"
)

# --- Pydantic Schemas ---
class RecommendationRequest(BaseModel):
    user_id: str
    num_recommendations: int = 5

class ProductItem(BaseModel):
    product_id: str
    category_name: Optional[str] = "N/A"
    product_name: Optional[str] = "N/A"

class RecommendationResponse(BaseModel):
    user_id: str
    status: str
    recommendations: List[ProductItem]


# --- Global Variables ---
PRODUCTS_DF = None


# --- Startup Event: Load Metadata ---
@app.on_event("startup")
def load_product_metadata():
    """
    Loads product data into memory when the FastAPI server starts.
    Ensure 'data/processed/products_metadata.csv' exists with columns:
    'product_id', 'product_category_name_english', and 'product_name'.
    """
    global PRODUCTS_DF
    try:
        # Update this path if your CSV is located elsewhere or named differently
        file_path = "data/processed/products_metadata.csv"
        PRODUCTS_DF = pd.read_csv(file_path).set_index("product_id")
        print("✅ Product metadata loaded successfully.")
    except FileNotFoundError:
        print("⚠️ Warning: products_metadata.csv not found. Reverting to UUIDs.")
        PRODUCTS_DF = pd.DataFrame()
    except Exception as e:
        print(f"⚠️ Error loading product metadata: {e}")
        PRODUCTS_DF = pd.DataFrame()


# --- Helper Function: Enrich Product IDs ---
def resolve_product_details(product_ids: List[str]) -> List[ProductItem]:
    """
    Takes a list of raw product UUIDs and maps them to actual category and names.
    """
    detailed_recs = []
    
    for pid in product_ids:
        if not PRODUCTS_DF.empty and pid in PRODUCTS_DF.index:
            row = PRODUCTS_DF.loc[pid]
            
            # Handle duplicate index entries if any exist
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            
            category = row.get("product_category_name_english", "General")
            
            # Olist dataset usually doesn't have a strict "product_name", 
            # so we fallback to a formatted ID or a specific column if you created one.
            name = row.get("product_name", f"Product {pid[:8]}")
        else:
            category = "Unknown Category"
            name = f"Product {pid[:8]}"
            
        detailed_recs.append(ProductItem(
            product_id=pid,
            category_name=str(category).title().replace("_", " "),
            product_name=str(name)
        ))
        
    return detailed_recs


# --- Mock ML Function (Remove this if you import your real function at the top) ---
def generate_lightfm_recommendations(user_id: str, num_recommendations: int):
    """
    Placeholder for your actual LightFM model inference.
    Returns: (list_of_product_ids, status_string)
    """
    # Dummy Olist product IDs for testing
    dummy_ids = [
        "154e7e31ebfa092203795c972e5804a6",
        "9dc1a7de274444849c219cff195d0b71",
        "79ce45dbc2ea29b22b5a261bbb7b7ee7",
        "27051f83715840aa06c4292128b97368",
        "3e5201fe0d1ba474b9b90152c83c706c"
    ]
    status = "personalized" # or "cold_start_fallback"
    return dummy_ids[:num_recommendations], status


# --- Core API Endpoint ---
@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendations(payload: RecommendationRequest):
    try:
        # 1. Fetch raw product IDs from the model
        raw_product_ids, status = generate_lightfm_recommendations(
            payload.user_id, 
            payload.num_recommendations
        )
        
        # 2. Enrich the raw IDs with actual category and names
        detailed_recommendations = resolve_product_details(raw_product_ids)
        
        # 3. Return the structured response
        return RecommendationResponse(
            user_id=payload.user_id,
            status=status,
            recommendations=detailed_recommendations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation Engine Error: {str(e)}")