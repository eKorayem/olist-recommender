from lightfm.data import Dataset
import pandas as pd
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

class SparseMatrixBuilder:
    def __init__(self):
        self.dataset = Dataset()
        logger.info("SparseMatrixBuilder initialized.")

    def build_sparse_matrices(self, interactions_df: pd.DataFrame, products_df: pd.DataFrame, artifacts_dir: str = "data/processed"):
        logger.info("Fitting LightFM dataset with unique users, items, and item features...")
        
        # 1. Handle missing categories and get unique features
        products_df['product_category_name'] = products_df['product_category_name'].fillna("unknown")
        unique_categories = products_df['product_category_name'].unique()
        
        # 2. Fit the dataset vocabulary, now including item_features
        self.dataset.fit(
            users=interactions_df['customer_unique_id'].unique(),
            items=interactions_df['product_id'].unique(),
            item_features=unique_categories
        )
        
        # 3. Build Interactions
        interactions_gen = zip(
            interactions_df['customer_unique_id'], 
            interactions_df['product_id'], 
            interactions_df['interaction_weight']
        )
        interactions, weights = self.dataset.build_interactions(interactions_gen)
        
        # 4. Build Item Features Matrix
        logger.info("Building item features matrix...")
        # Create a list of tuples: (product_id, [list_of_features])
        item_features_gen = (
            (row['product_id'], [row['product_category_name']]) 
            for index, row in products_df.iterrows() 
            if row['product_id'] in interactions_df['product_id'].values
        )
        
        item_features_matrix = self.dataset.build_item_features(item_features_gen)
        
        # 5. Extract mappings
        user_id_map, _, item_id_map, item_feature_map = self.dataset.mapping()
        
        # 6. Save mappings
        artifacts_path = Path(artifacts_dir)
        artifacts_path.mkdir(parents=True, exist_ok=True)
        
        with open(artifacts_path / "user_map.pkl", "wb") as f:
            pickle.dump(user_id_map, f)
        with open(artifacts_path / "item_map.pkl", "wb") as f:
            pickle.dump(item_id_map, f)
            
        logger.info(f"Matrices built. Item features shape: {item_features_matrix.shape}")
        
        return interactions, weights, item_features_matrix, user_id_map, item_id_map