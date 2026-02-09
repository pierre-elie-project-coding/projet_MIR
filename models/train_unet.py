import torch
import torch.nn as nn
from models.unet import Unet
from data_process.unet_preprocess_data import preprocess_data_for_unet
from torchmetrics.classification import MulticlassF1Score
from data_process.process_data import fetch_data_for_training
from utils.parse_config import get_config
from utils.get_loss_and_optimizer import get_loss,get_optimizer
import os


def train_unet(stop: int | None = None):

    config = get_config()
    sliding_window_size = config["model"]["unet"]["sliding_window_size"]
    optimizer_name = config["training"]["unet"]["optimizer"]
    loss_name = config["training"]["unet"]["loss"]
    learning_rate = config["training"]["unet"]["lr"]
    epochs = config["training"]["unet"]["epoch"]
    precision = config["model"]["unet"]["precision"]
    return_weights_for_loss = config["training"]["unet"]["loss_weight"]
    padding = config["model"]["unet"]["padding"]

    # Fetching data
    inputs, labels, weights = fetch_data_for_training(stop=stop,return_weights_for_loss=return_weights_for_loss)

    train_dataloader, test_dataloader = preprocess_data_for_unet(
        input_tensor=inputs, labels=labels
    )

    # Device
    device = (
        torch.accelerator.current_accelerator().type  # type: ignore
        if torch.accelerator.is_available()
        else "cpu"
    )

    print(f"Using {device} device")
    print(f"Training model in {precision} precision")

    # Instance of the mlp class
    model = Unet(kernel_size=3,pool_kernel_size=2,channels=[64,128,256,512,1024],kernel_size_upconv=2,padding=padding).to(device)

    if precision == "half":    
        model.to(torch.half)

    # TODO tune both of them to improve
    # Loss function 
    loss_params = {"weight":weights}
    print(f"Loss params : {loss_params}")
    loss_fn = get_loss(loss=loss_name,**loss_params)
    # Optimizer
    optim_params = {"lr":learning_rate}
    optimizer = get_optimizer(optim=optimizer_name,model_params=model.parameters(),**optim_params)

    # Training loop : TODO tuning epoch number
    epochs_list = []
    loss_list = []
    accuracy_list = []
    f1score_list=  []
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        epochs_list.append(t+1)
        train(train_dataloader, model, loss_fn, optimizer,sliding_window_size=sliding_window_size ,device=device)
        loss_current ,accuracy_current ,f1score_current = test(test_dataloader, model, loss_fn, sliding_window_size=sliding_window_size, device=device)
        loss_list.append(loss_current)
        accuracy_list.append(accuracy_current)
        f1score_list.append(f1score_current)
    print("Done!")

    os.makedirs("weights/unet", exist_ok=True)
    torch.save(model.state_dict(), "weights/unet/unet.pth")
    print("Saved PyTorch Model State to weights/unet/unet.pth")

    return epochs_list ,loss_list ,accuracy_list ,f1score_list 

def train(dataloader, model, loss_fn, optimizer, sliding_window_size:int,device: str = "cpu"):
    loss_list = []
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        X = X.reshape(-1,1,sliding_window_size) # reshaping to have 1 channel
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()  # remise a zero
        loss.backward()  # calcul des gradients
        optimizer.step()  # mise a jour des poids

        if batch % 100 == 0:
            loss_value, current = loss.item(), (batch + 1) * len(X)
            loss_list.append(loss_value)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")
    return loss_list


def test(dataloader, model, loss_fn, sliding_window_size:int ,device: str = "cpu"):
    num_batches = len(dataloader)
    model.eval()
    size = 0
    test_loss, correct = 0, 0
    metric = MulticlassF1Score(num_classes=6).to(device=device)
    f1score = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X = X.reshape(-1,1,sliding_window_size) # reshaping to have 1 channel
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            size += y.numel()
            f1score += metric(pred, y)
    test_loss /= num_batches  # random loss : 1.79
    f1score_mean = f1score / num_batches
    print(f"CorrecT : {correct}")
    print(f"size : {size}")
    correct /= size
    print(
        f"Test Error: \n F1-score : {(100 * f1score_mean):>0.1f}% Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
    return test_loss,correct,f1score_mean # correct is the accuracy
