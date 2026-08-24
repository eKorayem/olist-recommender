import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import our custom modules
from src.data.data_loader import OlistDataLoader
from src.features.build_features import InteractionBuilder, SparseMatrixBuilder
from src.models.train_model import RecommenderModel
from src.models.inference import RecommenderInference

logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="E-commerce Recommender API",
    description="Scalable inference engine for the Olist dataset.",
    version="1.0.0"
)

# Global variables to hold our model state in memory
inference_engine = None
popular_items = []

# Define the expected JSON input schema
class UserRequest(BaseModel):
    user_id: str
    num_recommendations: int = 5

@app.on_event("startup")
async def startup_event():
    """
    Runs automatically when the API starts. 
    It trains the model once and loads it into memory so it can serve requests instantly.
    """
    global inference_engine, popular_items
    logger.info("Starting up API and warming the model cache...")
    
    # 1. Load and process data
    loader = OlistDataLoader(raw_data_path="data/raw/")
    tables = loader.load_core_tables()
    interactions_df = InteractionBuilder().build_implicit_interactions(tables)
    
    # 2. Compute popular items for Cold Start fallback
    popular_items = interactions_df['product_id'].value_counts().head(5).index.tolist()
    
    # 3. Build Matrices and Train
    matrix_builder = SparseMatrixBuilder()
    interactions, weights, user_map, item_map = matrix_builder.build_sparse_matrices(interactions_df)
    
    recommender = RecommenderModel(no_components=30)
    trained_model = recommender.train_and_evaluate(interactions)
    
    # 4. Initialize Inference Engine
    inference_engine = RecommenderInference(trained_model, user_map, item_map)
    logger.info("API is ready to serve predictions.")

@app.post("/recommend")
async def get_recommendations(request: UserRequest):
    """
    Endpoint that accepts a user ID and returns recommendations.
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model is still initializing. Please wait.")
        
    # Generate recommendations
    recs = inference_engine.recommend_for_user(request.user_id, request.num_recommendations)
    
    # Cold Start Fallback
    if not recs:
        logger.info(f"Cold Start Fallback triggered for user {request.user_id}")
        return {
            "user_id": request.user_id,
            "status": "cold_start_fallback",
            "recommendations": popular_items[:request.num_recommendations]
        }
        
    return {
        "user_id": request.user_id,
        "status": "personalized",
        "recommendations": recs
    }