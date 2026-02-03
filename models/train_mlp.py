import torch
import torch.nn as nn
from models.mlp import MlpSlidingWindow
from data_process.mlp_preprocess_data import preprocess_data_for_mlp
from data_process.process_data import fetch_data_for_training, tensor_to_dataloader
from models.train import train
from models.test import test

def train_mlp():

    # Fetching data
    batch_size = 64
    inputs,labels = fetch_data_for_training()
    input_tensor,label_tensor = preprocess_data_for_mlp(input_tensor=inputs,labels=labels,sliding_window_size=101)
    train_dataloader,test_dataloader = tensor_to_dataloader(input_tensor=input_tensor,labels=label_tensor,batch_size=batch_size,with_split=0.8)

    # Device
    device = (
        torch.accelerator.current_accelerator().type # type: ignore
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Instance of the mlp class
    model = MlpSlidingWindow().to(device)
    print(model)

    # Loss function + optimizer : TODO tune both of them to improve
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Training loop : TODO adding batch normalization + tuning epoch number
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")
