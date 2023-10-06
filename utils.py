import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import mlflow
import copy
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import hopsworks
from config import Config
config = Config()
def parse(x:str):
 return datetime.strptime(x, '%Y %m %d %H')


def read_csv(path):
    return pd.read_csv(path,delimiter=";")

def preprocess(df):
    df['HourUTC_1'] = pd.to_datetime(df['HourUTC'])
    df['Hour'] = df['HourUTC_1'].dt.hour
    df['Year'] = df['HourUTC_1'].dt.year
    df['Month'] = df['HourUTC_1'].dt.month
    df['Day'] = df['HourUTC_1'].dt.day
    df['Day_of_week'] = df['HourUTC_1'].dt.dayofweek
    df['ConsumerType_DE35'] = df['ConsumerType_DE35'].astype(str)
    df['ConsumerType_Area'] = df['PriceArea'].str.cat(df['ConsumerType_DE35'], sep='_')
    df = df[["HourUTC","Year","Month","Day","Hour","Day_of_week","ConsumerType_Area","TotalCon"]]
    return df


def feature_extraction(df):
    encoder = LabelEncoder()
    df['ConsumerType_Area'] = encoder.fit_transform(df['ConsumerType_Area'])
    df = df.sort_values(by=["ConsumerType_Area",'HourUTC'])
    
    for hour in range(0,24):
        df['TotalCon'+str(hour+1)] = df.groupby(["ConsumerType_Area"])['TotalCon'].shift(hour+1)
        
    df = df.dropna()

    for hour in range(0,24):
        df['TotalCon'+str(hour+1)] = df['TotalCon'+str(hour+1)].astype(int)
        
    return df


def split_data(df):
    train_size = int(len(df) * 0.9)
    train, test = df[:train_size], df[train_size:]
    return train, test



def create_dataset(df,hour_look_back):
    features_1 = [f"TotalCon{i}" for i in range(1, hour_look_back + 1)]
    features_2 = ["Day_of_week","ConsumerType_Area","Year","Month","Day","Hour"]
    features = features_2 + features_1 
    target = "TotalCon"
    X = df[features].values.astype(dtype=float)
    y = df[target].values.astype(dtype=float)
    return X, y


def get_data_from_hopswork(start_time,end_time,version):
    features_1 = [f"totalcon{i}" for i in range(1, 24 + 1)]
    features_2 = ["day_of_week","consumertype_area","year","month","day","hour"]
    target = "totalcon"
    features = features_2 + features_1
    project = login_hopswork()
    fs = get_fs(project)
    fg = get_fg(fs,"tracking_energy_consumertype_per_hour",version=version)
    query = fg.select_all().filter(
                (fg['hourutc'] >= start_time) &
                (fg['hourutc'] <= end_time)
            ).read()
    X = query[features].values.astype(dtype=float)
    y = query[target].values.astype(dtype=float)
    return X,y


def login_hopswork():
    project = hopsworks.login()
    return project


def get_fs(project):
    return project.get_feature_store()


def get_fg(fs,fg_name,version):
    return fs.get_feature_group(fg_name,version=version)


def visualize_predictions(model, data_loader, name_figure, path_save_plot):
    # model.eval()
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            # Forward pass to get predictions
            outputs = model.predict(x_batch.numpy())
            predictions = abs(outputs)  # Assuming predictions are 1D

            # Ground truth values (y_batch)
            ground_truth = y_batch.numpy()

            # Visualize the results using a line plot
            plt.figure(figsize=(10, 6))
            plt.plot(ground_truth, label="Ground Truth", marker="o", linestyle="-")
            plt.plot(predictions, label="Predictions", marker="o", linestyle="--")
            plt.xlabel("Time Step")
            plt.ylabel("Value")
            plt.title(f"Model Predictions vs Ground Truth")
            plt.legend()
            plt.show()
            plt.savefig(f"{path_save_plot}/{name_figure}.png")
            break  # Visualize only the first batch of data