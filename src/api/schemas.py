from pydantic import BaseModel
from typing import List, Optional

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