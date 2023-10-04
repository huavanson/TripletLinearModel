
class Config:
    def __init__(self):
        self.hour_look_back = 24
        self.path = "D:\data_for_project.csv"
        self.batch_size = 32
        self.path_save_ckp = "application/checkpoints"
        self.feature_size = 25
        self.num_epochs = 10
        self.lr = 0.001
        self.name_figure = "visualize_predict_first_batch"
        self.path_save_plot = "D:"



