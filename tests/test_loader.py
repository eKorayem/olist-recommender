import pytest
import pandas as pd
from src.data.data_loader import OlistDataLoader
from src.features.build_features import InteractionBuilder

def test_data_loader():
    loader = OlistDataLoader(raw_data_path="data/raw/")
    tables = loader.load_core_tables()
    
    assert "orders" in tables
    assert "items" in tables
    assert "products" in tables
    assert "customers" in tables
    assert isinstance(tables["orders"], pd.DataFrame)

def test_interaction_builder():
    loader = OlistDataLoader(raw_data_path="data/raw/")
    tables = loader.load_core_tables()
    
    builder = InteractionBuilder()
    interactions_df = builder.build_interactions(
        orders=tables["orders"],
        customers=tables["customers"],
        items=tables["items"],
        output_path="data/processed/interactions.parquet"
    )
    
    assert isinstance(interactions_df, pd.DataFrame)
    assert "customer_unique_id" in interactions_df.columns
    assert "product_id" in interactions_df.columns
    assert "interaction_weight" in interactions_df.columns
    assert len(interactions_df) > 0