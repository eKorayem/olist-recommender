import pandas as pd
import logging

logger = logging.getLogger(__name__)

class InteractionBuilder:
    """
    Transforms raw Olist tables into user-item interaction matrices.
    """
    def __init__(self):
        logger.info("InteractionBuilder initialized.")

    def build_implicit_interactions(self, tables: dict) -> pd.DataFrame:
        """
        Merges orders, customers, and items to create a user-item interaction dataframe.
        """
        logger.info("Building implicit interactions dataframe...")
        
        # Step 1: Merge orders with customers to get the REAL user ID (customer_unique_id)
        orders_customers = pd.merge(
            tables["orders"], 
            tables["customers"], 
            on="customer_id", 
            how="inner"
        )
        
        # Step 2: Merge the result with items to see WHAT they bought
        full_merge = pd.merge(
            orders_customers, 
            tables["items"], 
            on="order_id", 
            how="inner"
        )
        
        # Step 3: Count how many times each user bought each item
        interactions = (
            full_merge.groupby(["customer_unique_id", "product_id"])
            .size()
            .reset_index(name="interaction_weight")
        )
        
        logger.info(f"Generated {len(interactions)} unique user-item interactions.")
        return interactions