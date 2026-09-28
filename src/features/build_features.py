import logging
import pickle
import pandas as pd
from pathlib import Path
from lightfm.data import Dataset

logger = logging.getLogger(__name__)

class InteractionBuilder:
    def __init__(self):
        logger.info("InteractionBuilder initialized.")

    def build_interactions(self, orders: pd.DataFrame, customers: pd.DataFrame, items: pd.DataFrame, output_path: str = "data/processed/interactions.parquet") -> pd.DataFrame:
        logger.info("Merging dataframes to build interactions...")
        
        # Merge orders to customers to get the customer_unique_id
        merged_df = orders.merge(customers, on='customer_id', how='inner')
        
        # Merge with items to get the product_id
        merged_df = merged_df.merge(items, on='order_id', how='inner')
        
        # Calculate interaction weights (number of times a user bought a specific product)
        interactions = (
            merged_df.groupby(['customer_unique_id', 'product_id'])
            .size()
            .reset_index(name='interaction_weight')
        )
        
        # Save the interaction table as a highly compressed Parquet file
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        interactions.to_parquet(path, index=False)
        
        logger.info(f"Built interactions table with {len(interactions)} rows. Saved to {output_path}")
        return interactions


class SparseMatrixBuilder:
    def __init__(self):
        self.dataset = Dataset()
        logger.info("SparseMatrixBuilder initialized.")

    def build_sparse_matrices(self, interactions_df: pd.DataFrame, products_df: pd.DataFrame, artifacts_dir: str = "data/processed"):
        logger.info("Fitting LightFM dataset with unique users, items, and item features...")
        
        products_df['product_category_name'] = products_df['product_category_name'].fillna("unknown")
        unique_categories = products_df['product_category_name'].unique()
        
        # Fit dataset vocabulary
        self.dataset.fit(
            users=interactions_df['customer_unique_id'].unique(),
            items=interactions_df['product_id'].unique(),
            item_features=unique_categories
        )
        
        # Build User-Item Interactions
        interactions_gen = zip(
            interactions_df['customer_unique_id'], 
            interactions_df['product_id'], 
            interactions_df['interaction_weight']
        )
        interactions, weights = self.dataset.build_interactions(interactions_gen)
        
        # Optimized: Vectorized filtering for item features
        logger.info("Building item features matrix...")
        valid_items = set(interactions_df['product_id'].unique())
        filtered_products = products_df[products_df['product_id'].isin(valid_items)]
        
        item_features_gen = zip(
            filtered_products['product_id'], 
            filtered_products['product_category_name'].apply(lambda cat: [cat])
        )
        item_features_matrix = self.dataset.build_item_features(item_features_gen)
        
        user_id_map, _, item_id_map, _ = self.dataset.mapping()
        
        artifacts_path = Path(artifacts_dir)
        artifacts_path.mkdir(parents=True, exist_ok=True)
        
        with open(artifacts_path / "user_map.pkl", "wb") as f:
            pickle.dump(user_id_map, f)
        with open(artifacts_path / "item_map.pkl", "wb") as f:
            pickle.dump(item_id_map, f)
            
        logger.info(f"Matrices built. Item features shape: {item_features_matrix.shape}")
        
        return interactions, weights, item_features_matrix, user_id_map, item_id_map