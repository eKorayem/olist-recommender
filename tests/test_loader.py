from src.data.data_loader import OlistDataLoader

# Make sure this path points to where your CSVs are!
loader = OlistDataLoader(raw_data_path="data/raw/") 
my_tables = loader.load_core_tables()

print(my_tables["orders"].shape)