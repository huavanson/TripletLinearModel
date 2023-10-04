from model import AirModel, AirModel2
from dataset import MyDataset
from utils import *
from config import Config

config = Config()

def train_process(train_loader, num_epochs, path_save_ckp):
    # Initialize the model
    model = AirModel(input_size=config.feature_size)
    # model = AirModel2()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Initialize MLflow
    mlflow.start_run()
    print("hello")
    best_val_loss = float("inf")
    check_loss = []
    best_model = copy.deepcopy(model)
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            # Forward pass
           
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss = torch.sqrt(loss)
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Print batch information if desired
            if (batch_idx + 1) % 10000 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        check_loss.append(avg_loss)

        # Save best checkpoint
        if avg_loss < best_val_loss:
            torch.save(
                model,
                f"{path_save_ckp}/best_model.pt",
            )
            best_model = copy.deepcopy(model)
            best_val_loss = avg_loss

        # Log metrics and parameters with MLflow
        mlflow.log_metric("loss", avg_loss, step=epoch + 1)
        mlflow.pytorch.autolog()
        mlflow.pytorch.log_model(model, "models")

        # Print epoch information
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # End MLflow run
    mlflow.end_run()
    return best_model

def main():
    
    df = read_csv(config.path)
    preprocessed_df = preprocess(df)
    feature_extractioned_df = feature_extraction(preprocessed_df)
    train, test = split_data(feature_extractioned_df)
    
    X_train, y_train = create_dataset(train, hour_look_back=config.hour_look_back)
    X_test, y_test = create_dataset(test, hour_look_back=config.hour_look_back)
    

    train_dataset = MyDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataset = MyDataset(X_test, y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=True)

    best_model = train_process(train_loader=train_loader,num_epochs=config.num_epochs,path_save_ckp=config.path_save_ckp)    
    
    visualize_predictions(model=best_model,data_loader=test_loader,name_figure=config.name_figure,path_save_plot=config.path_save_plot)

if __name__ == '__main__':
    main()