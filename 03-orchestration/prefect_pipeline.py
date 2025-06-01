import pandas as pd
import prefect
import mlflow

data_link = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"

def read_data(url):

    df = pd.read_parquet(url)
    num_records = df.shape[0]
    print(f"The number of records is {num_records}.")
    return 

def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df

if __name__ == '__main__':
    read_data(data_link)