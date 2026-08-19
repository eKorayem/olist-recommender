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