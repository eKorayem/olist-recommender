## export_code_to_md.py

```python
import os
import json

ROOT_DIR = "."
OUTPUT_FILE = "project_code.md"

IGNORE_DIRS = {"__pycache__", ".git", ".vscode", ".claude", ".ipynb_checkpoints"}

with open(OUTPUT_FILE, "w", encoding="utf-8") as md_file:
    for root, dirs, files in os.walk(ROOT_DIR):
        # Skip unwanted directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            filepath = os.path.join(root, file)
            rel_path = os.path.relpath(filepath, ROOT_DIR)

            # Handle standard Python and .env files
            if file.endswith(".py") or file.startswith(".env"):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()

                    md_file.write(f"## {rel_path}\n\n")
                    md_file.write("```python\n")
                    md_file.write(content)
                    md_file.write("\n```\n\n")

                except Exception as e:
                    print(f"Skipping {rel_path}: {e}")

            # Handle Jupyter Notebook files
            elif file.endswith(".ipynb"):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        notebook = json.load(f)
                    
                    md_file.write(f"## {rel_path}\n\n")
                    md_file.write("```python\n")
                    
                    # Loop through cells and only extract source from code cells
                    for cell in notebook.get("cells", []):
                        if cell.get("cell_type") == "code":
                            # 'source' is a list of strings (lines of code)
                            source_code = "".join(cell.get("source", []))
                            if source_code.strip(): # Only write non-empty cells
                                md_file.write(source_code)
                                md_file.write("\n\n# --- End of Cell ---\n\n")
                                
                    md_file.write("\n```\n\n")

                except Exception as e:
                    print(f"Skipping {rel_path}: {e}")

print(f"Done Exported all .py, .env, and .ipynb code to {OUTPUT_FILE}")
```

## notebooks/01_eda_olist.ipynb

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- End of Cell ---

raw_data_path = "../data/raw/"
orders = pd.read_csv(raw_data_path + "olist_orders_dataset.csv")
items = pd.read_csv(raw_data_path + "olist_order_items_dataset.csv")
customers = pd.read_csv(raw_data_path + "olist_customers_dataset.csv")
products = pd.read_csv(raw_data_path + "olist_products_dataset.csv")

# --- End of Cell ---

data = {
    "Orders" : orders,
    "Items" : items,
    "Customers" : customers,
    "Products" : products
}

for k, d in data.items():
    print(k)
    print(d.columns)

# --- End of Cell ---

merged = orders.merge(customers, on="customer_id").merge(items, on="order_id")
num_users = merged['customer_unique_id'].nunique()
num_items = merged['product_id'].nunique()
total_interactions = len(merged)
possible_interactions = num_users * num_items
sparsity = 1 - (total_interactions / possible_interactions)

print(f"Total Unique Users: {num_users}")
print(f"Total Unique Items: {num_items}")
print(f"Total Purchases (Interactions): {total_interactions}")
print(f"Matrix Sparsity: {sparsity * 100:.4f}%")

# --- End of Cell ---

# 1. Count how many times each product was bought
item_popularity = merged['product_id'].value_counts()

# 2. Plot the Long Tail
plt.figure(figsize=(12, 6))
plt.plot(item_popularity.values, color='blue', linewidth=2)
plt.fill_between(range(len(item_popularity)), item_popularity.values, color='blue', alpha=0.3)
plt.title('The Long Tail of Olist Product Popularity', fontsize=16)
plt.xlabel('Products (Ranked from most to least popular)', fontsize=12)
plt.ylabel('Number of Purchases', fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

# 3. Calculate the Pareto Principle (80/20 Rule)
cumulative_sales = item_popularity.cumsum()
total_sales = item_popularity.sum()
cumulative_percentage = cumulative_sales / total_sales

# Find how many items it takes to reach 80% of total sales
items_80_percent = len(cumulative_percentage[cumulative_percentage <= 0.8])
percent_catalog_80_sales = (items_80_percent / len(item_popularity)) * 100

print(f"Total Unique Items: {len(item_popularity)}")
print(f"Top {items_80_percent} items ({percent_catalog_80_sales:.2f}% of the catalog) account for 80% of all sales.")
print(f"The remaining {len(item_popularity) - items_80_percent} items make up the massive 'Long Tail'.")

# --- End of Cell ---


```

## notebooks/play_with_data.ipynb

```python
import pandas as pd
import os

# Define the relative path from the notebooks folder to the raw data folder
RAW_DATA_DIR = '../data/raw/'

# Load the datasets into Pandas DataFrames
customers_df = pd.read_csv(f"{RAW_DATA_DIR}olist_customers_dataset.csv")
orders_df = pd.read_csv(f"{RAW_DATA_DIR}olist_orders_dataset.csv")
products_df = pd.read_csv(f"{RAW_DATA_DIR}olist_products_dataset.csv")
geolocation_df = pd.read_csv(f"{RAW_DATA_DIR}olist_geolocation_dataset.csv")
order_payments_df = pd.read_csv(f"{RAW_DATA_DIR}olist_order_payments_dataset.csv")
order_reviews_df = pd.read_csv(f"{RAW_DATA_DIR}olist_order_reviews_dataset.csv")
sellers_olist_sellers_dataset = pd.read_csv(f"{RAW_DATA_DIR}olist_sellers_dataset.csv")


# --- End of Cell ---

products_df.head()

# --- End of Cell ---


```

## src/__init__.py

```python

```

## src/api/main.py

```python
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
```

## src/api/__init__.py

```python

```

## src/data/data_loader.py

```python
import pandas as pd
import logging
from pathlib import Path

# Configure production-grade logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OlistDataLoader:
    """
    Encapsulates data ingestion logic for the Olist E-commerce dataset.
    """
    def __init__(self, raw_data_path: str):
        # We use pathlib because it handles Windows/Linux path slashes automatically
        self.raw_data_path = Path(raw_data_path)
        logger.info(f"OlistDataLoader initialized. Pointing to: {self.raw_data_path}")

    def load_core_tables(self) -> dict:
        """
        Loads the essential CSV files into memory.
        
        Returns:
            dict: A dictionary containing pandas DataFrames.
        """
        logger.info("Loading core Olist tables...")
        
        try:
            # We are loading just the files we need for now
            tables = {
                "orders": pd.read_csv(self.raw_data_path / "olist_orders_dataset.csv"),
                "items": pd.read_csv(self.raw_data_path / "olist_order_items_dataset.csv"),
                "products": pd.read_csv(self.raw_data_path / "olist_products_dataset.csv"),
                "customers": pd.read_csv(self.raw_data_path / "olist_customers_dataset.csv"),
            }
            logger.info("Successfully loaded all core tables.")
            return tables
            
        except FileNotFoundError as e:
            logger.error(f"Critical Error: Could not find data file. Ensure you extracted the Kaggle zip to {self.raw_data_path}")
            raise e
```

## src/data/__init__.py

```python

```

## src/features/build_features.py

```python
import pandas as pd
import logging
from pathlib import Path

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
    """
    Converts pandas interaction DataFrames into scipy sparse matrices required by LightFM.
    """
    def __init__(self):
        self.dataset = Dataset()
        logger.info("SparseMatrixBuilder initialized.")

    def build_sparse_matrices(self, interactions_df: pd.DataFrame):
        """
        Fits the LightFM dataset and builds the interaction and weight matrices.
        """
        logger.info("Fitting LightFM dataset with unique users and items...")
        
        # 1. Fit the dataset vocabulary
        self.dataset.fit(
            users=interactions_df['customer_unique_id'].unique(),
            items=interactions_df['product_id'].unique()
        )
        
        logger.info("Building sparse COO matrices...")
        
        # 2. Create an iterable of (user, item, weight) tuples
        interactions_gen = zip(
            interactions_df['customer_unique_id'], 
            interactions_df['product_id'], 
            interactions_df['interaction_weight']
        )
        
        # 3. Build the matrices
        interactions, weights = self.dataset.build_interactions(interactions_gen)
        
        # 4. Extract the internal mapping dictionaries
        user_id_map, _, item_id_map, _ = self.dataset.mapping()
        
        logger.info(f"Interaction matrix shape: {interactions.shape}")
        return interactions, weights, user_id_map, item_id_map
```

## src/features/__init__.py

```python

```

## src/models/inference.py

```python
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
```

## src/models/train_model.py

```python
import logging
import mlflow
import pickle
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score

logger = logging.getLogger(__name__)

class RecommenderModel:
    """
    Handles the training, evaluation, and tracking of the LightFM model.
    """
    def __init__(self, no_components: int = 30, learning_rate: float = 0.05, loss: str = 'warp'):
        self.no_components = no_components
        self.learning_rate = learning_rate
        self.loss = loss
        
        self.model = LightFM(
            no_components=self.no_components,
            learning_rate=self.learning_rate,
            loss=self.loss,
            random_state=42
        )
        logger.info(f"RecommenderModel initialized with {self.loss} loss and {self.no_components} components.")

    def train_and_evaluate(self, interactions, epochs: int = 10, num_threads: int = 4):
        """
        Trains the model, evaluates it, and logs everything to MLflow.
        """
        logger.info(f"Starting MLflow run. Training for {epochs} epochs...")
        mlflow.set_experiment("Olist_Recommendation_Engine")
        
        with mlflow.start_run():
            # 1. Log Hyperparameters
            mlflow.log_param("no_components", self.no_components)
            mlflow.log_param("learning_rate", self.learning_rate)
            mlflow.log_param("loss", self.loss)
            mlflow.log_param("epochs", epochs)
            
            # 2. Train the model
            self.model.fit(
                interactions,
                epochs=epochs,
                num_threads=num_threads,
                verbose=True
            )
            
            # 3. Evaluate the model
            logger.info("Evaluating model metrics (this may take a moment)...")
            train_precision = precision_at_k(self.model, interactions, k=10, num_threads=num_threads).mean()
            train_auc = auc_score(self.model, interactions, num_threads=num_threads).mean()
            
            logger.info(f"Precision@10: {train_precision:.4f} | AUC: {train_auc:.4f}")
            
            # 4. Log Metrics to MLflow
            mlflow.log_metric("precision_at_10", train_precision)
            mlflow.log_metric("auc", train_auc)
            
            # 5. Save the trained model artifact
            with open("model_artifact.pkl", "wb") as f:
                pickle.dump(self.model, f)
            mlflow.log_artifact("model_artifact.pkl")
            
            logger.info("Run successfully logged to MLflow.")
            
        return self.model
```

## src/models/__init__.py

```python

```

## tests/test_loader.py

```python
from src.data.data_loader import OlistDataLoader
from src.features.build_features import InteractionBuilder, SparseMatrixBuilder
from src.models.train_model import RecommenderModel # NEW IMPORT

# 1. Load Data
loader = OlistDataLoader(raw_data_path="data/raw/") 
my_tables = loader.load_core_tables()

# 2. Build Interactions DF
builder = InteractionBuilder()
interactions_df = builder.build_implicit_interactions(my_tables)

# 3. Build Sparse Matrices
matrix_builder = SparseMatrixBuilder()
interactions, weights, user_map, item_map = matrix_builder.build_sparse_matrices(interactions_df)

# 4. Train Model (NEW CODE)
recommender = RecommenderModel(no_components=30, learning_rate=0.05)
trained_model = recommender.train_and_evaluate(interactions, epochs=10)

from src.models.inference import RecommenderInference

# Initialize the Inference Engine
inferencer = RecommenderInference(trained_model, user_map, item_map)

# Pick the very first user from our interactions dataframe
test_user = interactions_df['customer_unique_id'].iloc[0]

# Get 5 recommendations for them
recs = inferencer.recommend_for_user(test_user, num_recommendations=5)

print(f"Recommendations for {test_user}:")
for r in recs:
    print(r)
```

