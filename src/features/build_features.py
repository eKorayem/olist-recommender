import pandas as pd
import logging
from pathlib import Path
import pickle
from lightfm.data import Dataset



logger = logging.getLogger(__name__)

class InteractionBuilder:
    """
    Transforms raw Olist tables into user-item interaction matrices and saves them.
    """
    def __init__(self):
        logger.info("InteractionBuilder initialized.")

    def build_implicit_interactions(self, tables: dict, save_path: str = "data/processed/interactions.parquet") -> pd.DataFrame:
        """
        Merges orders, customers, and items, and saves the output as a Parquet file.
        """
        logger.info("Building implicit interactions dataframe...")
        
        # Step 1: Merge orders with customers
        orders_customers = pd.merge(
            tables["orders"], 
            tables["customers"], 
            on="customer_id", 
            how="inner"
        )
        
        # Step 2: Merge with items
        full_merge = pd.merge(
            orders_customers, 
            tables["items"], 
            on="order_id", 
            how="inner"
        )
        
        # Step 3: Count interactions
        interactions = (
            full_merge.groupby(["customer_unique_id", "product_id"])
            .size()
            .reset_index(name="interaction_weight")
        )
        
        logger.info(f"Generated {len(interactions)} unique user-item interactions.")
        
        # Step 4: Save to the processed directory
        if save_path:
            path_obj = Path(save_path)
            # This automatically creates the data/processed/ folder if it doesn't exist
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as highly-compressed Parquet
            interactions.to_parquet(path_obj, index=False)
            logger.info(f"Interactions successfully saved to {path_obj}")
            
        return interactions


from lightfm.data import Dataset

class SparseMatrixBuilder:
    def __init__(self):
        self.dataset = Dataset()
        logger.info("SparseMatrixBuilder initialized.")

    def build_sparse_matrices(self, interactions_df: pd.DataFrame, artifacts_dir: str = "data/processed"):
        logger.info("Fitting LightFM dataset with unique users and items...")
        
        self.dataset.fit(
            users=interactions_df['customer_unique_id'].unique(),
            items=interactions_df['product_id'].unique()
        )
        
        interactions_gen = zip(
            interactions_df['customer_unique_id'], 
            interactions_df['product_id'], 
            interactions_df['interaction_weight']
        )
        
        interactions, weights = self.dataset.build_interactions(interactions_gen)
        user_id_map, _, item_id_map, _ = self.dataset.mapping()
        
        # Save mappings to disk
        artifacts_path = Path(artifacts_dir)
        artifacts_path.mkdir(parents=True, exist_ok=True)
        
        with open(artifacts_path / "user_map.pkl", "wb") as f:
            pickle.dump(user_id_map, f)
        with open(artifacts_path / "item_map.pkl", "wb") as f:
            pickle.dump(item_id_map, f)
            
        logger.info(f"Mappings successfully saved to {artifacts_path}")
        return interactions, weights, user_id_map, item_id_map