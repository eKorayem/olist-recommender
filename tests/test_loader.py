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