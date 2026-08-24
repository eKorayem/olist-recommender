import logging
import numpy as np

logger = logging.getLogger(__name__)

class RecommenderInference:
    """Handles making predictions using the trained LightFM model."""
    
    def __init__(self, model, user_id_map: dict, item_id_map: dict):
        self.model = model
        self.user_id_map = user_id_map
        self.item_id_map = item_id_map
        
        # Create a reverse mapping (Integer ID -> String ID) to return real product IDs
        self.reverse_item_map = {v: k for k, v in self.item_id_map.items()}
        logger.info("RecommenderInference initialized with mapping dictionaries.")

    def recommend_for_user(self, customer_unique_id: str, num_recommendations: int = 5) -> list:
        """Generates top N item recommendations for a known user."""
        
        # 1. Cold Start Check
        if customer_unique_id not in self.user_id_map:
            logger.warning(f"User {customer_unique_id} not found. Cold start triggered.")
            # We will build the Cold Start fallback logic later!
            return []
            
        # 2. Get internal integer IDs
        internal_user_id = self.user_id_map[customer_unique_id]
        n_items = len(self.item_id_map)
        item_indices = np.arange(n_items)
        
        # 3. Predict and rank
        scores = self.model.predict(user_ids=internal_user_id, item_ids=item_indices)
        top_item_indices = np.argsort(-scores)[:num_recommendations]
        
        # 4. Map integers back to real-world product IDs
        return [self.reverse_item_map[i] for i in top_item_indices]