import hopsworks

class Config:
    def __init__(self):
        self.hour_look_back = 24
        self.path = "D:\data_for_project.csv"
        self.batch_size = 32
        self.path_save_ckp = "application/checkpoints"
        self.feature_size = 30
        self.num_epochs = 10
        self.lr = 0.0001
        self.name_figure = "visualize_predict_first_batch"
        self.path_save_plot = "D:\project_cloud_computing\plot"
        self.loaded_model = 'runs:/1f81ce9a3e414e56a9f1d0ef89cab7da/models'
        self.fg_name = "tracking_energy_consumertype_per_hour"
        self.model_name="TripletLinearModel"
        self.model_version=2
        
        

