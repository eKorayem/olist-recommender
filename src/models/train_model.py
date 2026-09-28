import logging
import pickle
from pathlib import Path
import mlflow
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score

logger = logging.getLogger(__name__)

class RecommenderModel:
    def __init__(self, no_components: int = 30, learning_rate: float = 0.05, loss: str = 'warp'):
        self.no_components = no_components
        self.learning_rate = learning_rate
        self.loss = loss
        self.model = LightFM(
            no_components=no_components,
            learning_rate=learning_rate,
            loss=loss,
            random_state=42
        )
        logger.info(f"RecommenderModel initialized with {loss} loss and {no_components} components.")

    def train_and_evaluate(
        self, 
        interactions, 
        item_features=None, 
        epochs: int = 30, 
        num_threads: int = 4, 
        output_dir: str = "data/processed"
    ):
        mlflow.set_experiment("Olist_Recommendation_Engine")
        
        with mlflow.start_run():
            mlflow.log_params({
                "no_components": self.no_components,
                "learning_rate": self.learning_rate,
                "loss": self.loss,
                "epochs": epochs
            })
            
            logger.info(f"Training LightFM model for {epochs} epochs...")
            self.model.fit(
                interactions,
                item_features=item_features,
                epochs=epochs,
                num_threads=num_threads,
                verbose=True
            )
            
            logger.info("Evaluating model performance...")
            train_precision = precision_at_k(
                self.model, 
                interactions, 
                item_features=item_features, 
                k=10, 
                num_threads=num_threads
            ).mean()
            
            train_auc = auc_score(
                self.model, 
                interactions, 
                item_features=item_features, 
                num_threads=num_threads
            ).mean()
            
            mlflow.log_metric("precision_at_10", train_precision)
            mlflow.log_metric("auc_score", train_auc)
            
            logger.info(f"Train Precision@10: {train_precision:.4f}")
            logger.info(f"Train AUC Score: {train_auc:.4f}")
            
            output_path = Path(output_dir) / "model_artifact.pkl"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                pickle.dump(self.model, f)
                
            logger.info(f"Saved trained model artifact to {output_path}")
            
            return self.model