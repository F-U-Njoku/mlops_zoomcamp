import pickle
from typing import Tuple

import mlflow
import mlflow.sklearn
import pandas as pd
from prefect import task, flow
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer


@task(retries=3, retry_delay_seconds=2)
def read_dataframe(filename: str) -> pd.DataFrame:
    """
    Read a Parquet file and perform simple transformations.
    """
    df = pd.read_parquet(filename)

    # Compute ride duration in minutes, then filter outliers
    df["duration"] = (
        df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    ).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    # Convert those location IDs to strings for one-hot encoding
    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    return df


@task(log_prints=True)
def train_model(
    df: pd.DataFrame, features: list[str], target: str
) -> Tuple[DictVectorizer, LinearRegression]:
    """
    Train a simple LinearRegression on df[features] → target.
    Also logs the model and parameters to MLflow.
    Returns the fitted DictVectorizer and the trained LinearRegression model.
    """
    dv = DictVectorizer()
    lr = LinearRegression()

    # Prepare feature matrix
    records = df[features].to_dict(orient="records")
    X_train = dv.fit_transform(records)
    y_train = df[target].values

    # Fit the model
    lr.fit(X_train, y_train)

    # Log parameters and model to MLflow (inside an active run)
    mlflow.log_param("features", features)
    mlflow.log_param("model_type", "LinearRegression")

    mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="model"
    )

    # Log the intercept as a metric just for visibility
    mlflow.log_metric("intercept", float(lr.intercept_))

    return dv, lr


@flow
def main_flow(
    data_link: str = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet",
    tracking_uri: str = "http://127.0.0.1:5000",
    exp_name: str = "orchestration_experiment",
):
    """
    Orchestrates:
      1. Reading data from a Parquet URL
      2. Training a LinearRegression model
      3. Logging & registering it in MLflow
    """
    # 1. Configure MLflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)

    # 2. Read & preprocess the DataFrame
    df = read_dataframe(data_link)

    # 3. Start an MLflow run to train and log the model
    with mlflow.start_run():
        features = ["PULocationID", "DOLocationID"]
        target = "duration"

        dv, model = train_model(df, features, target)

        # After logging the model under artifact_path="model",
        # register it under the given name.
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"

        # Change “linear-reg-model” to whatever you want to name your registered model
        registered_model_name = "linear-reg-model"
        mlflow.register_model(model_uri, registered_model_name)

    print(f"Finished training and registering run_id={run_id} as '{registered_model_name}'.")


if __name__ == "__main__":
    main_flow()