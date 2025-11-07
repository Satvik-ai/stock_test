"""
Train a Decision Tree classifier using CSV data, log metrics and model to MLflow.
Fixed for Vertex AI Workbench compatibility.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import joblib
import argparse
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from feast import FeatureStore
import gc
import traceback
import psutil


def main(n_estimators: int, max_depth: int, random_state: int, version: str, stratify: str = "NO"):
    print("="*50)
    print("Starting training script...")
    print(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}, version={version}")
    print("="*50)
    
    try:
        # --------------------------
        # Setup MLflow (with fallback)
        # --------------------------
        print("\n[1/6] Setting up MLflow...")
        try:
            mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8100")
            print(f"MLflow tracking URI: {mlflow_uri}")
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment("Stock_RF_Classification")
            use_mlflow = True
            print("✅ MLflow setup successful")
        except Exception as mlflow_error:
            print(f"⚠️  MLflow setup failed: {mlflow_error}")
            print("Continuing without MLflow logging...")
            use_mlflow = False

        if use_mlflow:
            with mlflow.start_run():
                mlflow.log_param("n_estimators", n_estimators)
                mlflow.log_param("max_depth", max_depth)
                mlflow.log_param("random_state", random_state)
                mlflow.log_param("stock_data_version", version)
                mlflow.log_param("feature_view", f"stock_features_{version}")
                mlflow.log_param("stratify", stratify)
                return _train_model(n_estimators, max_depth, random_state, version, stratify, use_mlflow)
        else:
            return _train_model(n_estimators, max_depth, random_state, version, stratify, use_mlflow)

    except Exception as e:
        print(f"\n❌ FATAL ERROR: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        raise


def _train_model(n_estimators, max_depth, random_state, version, stratify, use_mlflow):
    """Separated training logic for cleaner error handling"""

    try:
        # --------------------------
        # Load Local Data
        # --------------------------
        print("\n[2/6] Loading local data...")
        local_parquet_data = "data/stock_data.parquet"
        print(f"Loading from: {local_parquet_data}")
        
        if not os.path.exists(local_parquet_data):
            raise FileNotFoundError(f"Data file not found: {local_parquet_data}")
        
        data = pd.read_parquet(local_parquet_data)
        print(f"✅ Data loaded. Shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")

        # --------------------------
        # Initialize Feast Feature Store
        # --------------------------
        print("\n[3/6] Initializing Feast Feature Store...")
        feast_repo_path = "feature_repo"
        
        if not os.path.exists(feast_repo_path):
            raise FileNotFoundError(f"Feast repo not found: {feast_repo_path}")
        
        try:
            store = FeatureStore(repo_path=feast_repo_path)
            print("✅ Feast store initialized")
        except Exception as feast_error:
            print(f"❌ Failed to initialize Feast: {feast_error}")
            raise

        # --------------------------
        # Fetch Features for Training
        # --------------------------
        print(f"Memory before fetch: {psutil.virtual_memory().available/1e9:.2f} GB")
        print("\n[4/6] Fetching features from Feast...")
        print(f"Requesting feature view: stock_features_{version}")
        
        try:
            training_df = store.get_historical_features(
                entity_df=data,
                features=[
                    f"stock_features_{version}:open",
                    f"stock_features_{version}:high",
                    f"stock_features_{version}:low",
                    f"stock_features_{version}:close",
                    f"stock_features_{version}:volume",
                    f"stock_features_{version}:ma_15_min",
                    f"stock_features_{version}:ma_60_min",
                    f"stock_features_{version}:rsi_14",
                    f"stock_features_{version}:target",
                ],
            ).to_df()
            print("✅ Features fetched successfully")
        except Exception as feature_error:
            print(f"❌ Failed to fetch features: {feature_error}")
            print("\nAvailable feature views:")
            try:
                for fv in store.list_feature_views():
                    print(f"  - {fv.name}")
            except:
                print("  (Could not list feature views)")
            raise
        
        training_df.dropna(inplace=True)
        print(f"Training data shape after dropna: {training_df.shape}")
        print(training_df.head())

        # --------------------------
        # Train Model
        # --------------------------
        print("\n[5/6] Training RandomForestClassifier...")
        feature_columns = ['open', 'high', 'low', 'close', 'volume', 'ma_15_min', 'ma_60_min', 'rsi_14']

        X = training_df[feature_columns]
        y = training_df["target"]

        if stratify.lower() == "yes":
            print("Using stratified split...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=random_state
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=random_state
            )

        # Clean up full dataset
        del X, y, training_df
        gc.collect()

        print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

        # Fixed: was using max_depth for n_estimators
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1  # Use all available cores
        )

        print("Fitting model...")
        model.fit(X_train, y_train)

        print("Making predictions...")
        predictions = model.predict(X_test)

        accuracy_score = metrics.accuracy_score(y_test, predictions)
        precision = metrics.precision_score(y_test, predictions, average='weighted', zero_division=0)
        recall = metrics.recall_score(y_test, predictions, average='weighted', zero_division=0)
        f1 = metrics.f1_score(y_test, predictions, average='weighted', zero_division=0)

        mlflow.log_metric("accuracy", accuracy_score)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        print(f"Accuracy: {accuracy_score:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")

        mlflow.set_tag("Training Info", "RandomForestClassifier Model for Stock data")

        # --------------------------
        # Save Model
        # --------------------------
        os.makedirs("artifacts", exist_ok=True)
        model_path = "artifacts/model.joblib"
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

        # Log model to MLflow
        signature = infer_signature(X_train, model.predict(X_train))
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="stock_model",
            signature=signature,
            input_example=X_train.iloc[:5],  # Just 5 examples
            registered_model_name="Stock-Classifier-RF",
        )

        print("✅ Training and logging complete!")
        return model_path

    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RandomForestClassifier on Stock dataset with MLflow logging")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--version", type=str, required=True, help="Feature view version, e.g., v1")
    parser.add_argument("--stratify", type=str, default="NO")
    args = parser.parse_args()

    main(args.n_estimators, args.max_depth, args.random_state, args.version, args.stratify)