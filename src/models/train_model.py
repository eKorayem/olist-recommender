import logging
import mlflow
from lightfm import LightFM

logger = logging.getLogger(__name__)

class RecommenderModel:
    """
    Handles the training and tracking of the LightFM recommendation model.
    """
    def __init__(self, no_components: int = 30, learning_rate: float = 0.05, loss: str = 'warp'):
        self.no_components = no_components
        self.learning_rate = learning_rate
        self.loss = loss
        
        # Initialize the LightFM model
        self.model = LightFM(
            no_components=self.no_components,
            learning_rate=self.learning_rate,
            loss=self.loss,
            random_state=42 # For reproducibility
        )
        logger.info(f"RecommenderModel initialized with {self.loss} loss and {self.no_components} components.")

    def train(self, interactions, epochs: int = 10):
        """
        Trains the model and logs the experiment to MLflow.
        """
        logger.info(f"Starting MLflow run. Training for {epochs} epochs...")
        
        # Set up MLflow tracking
        mlflow.set_experiment("Olist_Recommendation_Engine")
        
        with mlflow.start_run():
            # 1. Log our hyperparameters to MLflow
            mlflow.log_param("no_components", self.no_components)
            mlflow.log_param("learning_rate", self.learning_rate)
            mlflow.log_param("loss", self.loss)
            mlflow.log_param("epochs", epochs)
            
            # 2. Train the model
            self.model.fit(
                interactions,
                epochs=epochs,
                num_threads=4, # Use multiple CPU cores for speed
                verbose=True
            )
            
            logger.info("Training complete.")
            # Note: We will add MLflow model saving and metric logging in the next step!
            
        return self.model