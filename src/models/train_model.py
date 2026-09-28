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
            logger.info("Evaluating model metrics...")
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