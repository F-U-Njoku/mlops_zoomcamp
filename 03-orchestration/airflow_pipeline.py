import mlflow
import datetime
import pandas as pd
from typing import Dict
from airflow.decorators import dag, task
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer


@dag(
    dag_id="orchestration",
    catchup=False,
    dagrun_timeout=datetime.timedelta(minutes=60),
    start_date=datetime.datetime(2025, 1, 1),
    tags=["mlops"]
)
def orchestration():

    @task(retries=3)
    def read(url: str) -> pd.DataFrame:
        df = pd.read_parquet(url)
        print(f"The number of loaded records is {df.shape[0]}")
        return df

    @task
    def process(df: pd.DataFrame) -> pd.DataFrame:
        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df.duration = df.duration.dt.total_seconds() / 60
        df = df[(df.duration >= 1) & (df.duration <= 60)]
        categorical = ['PULocationID', 'DOLocationID']
        df[categorical] = df[categorical].astype(str)
        print(f"The number of clean records is {df.shape[0]}")
        return df

    @task
    def train(df: pd.DataFrame, features: list[str], target: str) -> Dict:
        dv = DictVectorizer()
        lr = LinearRegression()
        records = df[features].to_dict(orient="records")
        X_train = dv.fit_transform(records)
        y_train = df[target].values
        lr.fit(X_train, y_train)
        return {"vec": dv, "model": lr}

    @task
    def register(tracking_uri: str, exp_name: str, artifacts: Dict) -> None:
        model = artifacts["model"]
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(exp_name)
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

    data_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"
    tracking_uri = "http://127.0.0.1:5000"
    exp_name = "orchestration_experiment"
    features = ["PULocationID", "DOLocationID"]
    target = "duration"

    df = read(data_url)
    clean_df = process(df)
    artifacts = train(clean_df, features, target)
    register(tracking_uri, exp_name, artifacts)


dag = orchestration()