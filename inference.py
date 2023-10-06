import mlflow
from config import Config
import pandas as pd
from utils import *
from dataset import MyDataset

config = Config()
logged_model = config.loaded_model
mlflow.set_tracking_uri("http://localhost:5002")

model = mlflow.pyfunc.load_model(model_uri=f"models:/{config.model_name}/{config.model_version}")

X_test,y_test = get_data_from_hopswork(start_time="2023-01-02",end_time="2023-01-10",version=1)

test_dataset = MyDataset(X_test, y_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)
visualize_predictions(model=model, data_loader=test_loader, name_figure=config.name_figure, path_save_plot=config.path_save_plot)


