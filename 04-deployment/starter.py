import uuid
import pickle
import argparse
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("year", type=int)
parser.add_argument("month", type=int)
args = parser.parse_args()

def load_model():
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    print(f"Artifacts loaded.")
    return dv, model

def read_data(filename, features):
    
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[features] = df[features].fillna(-1).astype('int').astype('str')

    print(f"File read from {filename}.")
    return df

def apply_model(df, features):
    dv, model = load_model()
    dicts = df[features].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print(f"Predictions done.")
    print(np.average(y_pred))
    return y_pred
    
def generate_uuids(n):
    ride_ids = []
    for i in range(n):
        ride_ids.append(str(uuid.uuid4()))

    return ride_ids

def return_result(n, predictions, output_file):
    generate_uuids(n)
    df_result = pd.DataFrame(data={"ride_id":generate_uuids(n), "prediction": predictions})
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
        )
    print(f"File written to {output_file}.")
    return  

def run():
    year = args.year
    month = args.month
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/yellow_tripdata/{year:04d}-{month:02d}.parquet'
    features = ['PULocationID', 'DOLocationID']

    df = read_data(input_file, features)
    predictions = apply_model(df, features)
    return_result(len(df), predictions, output_file)

if __name__ == '__main__':
    run()


