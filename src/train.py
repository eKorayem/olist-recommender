import logging
import pandas as pd
from pathlib import Path

from src.data.data_loader import OlistDataLoader 
from src.features.build_features import InteractionBuilder, SparseMatrixBuilder
from src.models.train_model import RecommenderModel

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Offline Recommendation Pipeline...")
    
    # 1. Load Raw Data (FIXED: Using correct parameter and method)
    loader = OlistDataLoader(raw_data_path="data/raw/")
    tables = loader.load_core_tables() 
    
    # 2. Build Interactions (Saves to parquet)
    builder = InteractionBuilder()
    interactions_df = builder.build_interactions(
        orders=tables["orders"], 
        customers=tables["customers"], 
        items=tables["items"], 
        output_path="data/processed/interactions.parquet"
    )
    
    # 3. Build Sparse Matrices & Artifacts (Saves mapping dicts)
    matrix_builder = SparseMatrixBuilder()
    interactions, weights, item_features_matrix, user_map, item_map = matrix_builder.build_sparse_matrices(
        interactions_df=interactions_df,
        products_df=tables["products"],
        artifacts_dir="data/processed"
    )
    
    # 4. Train Model with MLflow tracking (Saves model.pkl)
    recommender = RecommenderModel(no_components=30, learning_rate=0.05, loss='warp')
    
    model = recommender.train_and_evaluate(
        interactions=interactions, 
        item_features=item_features_matrix,
        epochs=30, 
        num_threads=4, 
        output_dir="data/processed"
    )
    
    logger.info("Pipeline Complete. Artifacts are ready for FastAPI inference.")

if __name__ == "__main__":
    main()