import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn as nn
from models.mlp import MlpSlidingWindow

def train_mlp():

    # Fetching data
    batch_size = 64 
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # Device
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" # type: ignore
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
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")