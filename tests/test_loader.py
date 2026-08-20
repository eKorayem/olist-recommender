from src.data.data_loader import OlistDataLoader
from src.features.build_features import InteractionBuilder

# 1. Load Data
loader = OlistDataLoader(raw_data_path="data/raw/") 
my_tables = loader.load_core_tables()

# 2. Build Interactions
builder = InteractionBuilder()
interactions_df = builder.build_implicit_interactions(my_tables)

# 3. Print a preview
print(interactions_df.head())
print(f"Total interactions: {len(interactions_df)}")