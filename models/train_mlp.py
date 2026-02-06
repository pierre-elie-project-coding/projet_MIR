import torch
import torch.nn as nn
from models.mlp import MlpSlidingWindow
from data_process.mlp_preprocess_data import preprocess_data_for_mlp
from torchmetrics.classification import MulticlassF1Score
from data_process.process_data import fetch_data_for_training
from utils.parse_config import get_config
from utils.get_loss_and_optimizer import get_loss,get_optimizer
from utils.statistics import get_targets_repartition, build_weights
import os


def train_mlp(stop: int | None = None):

    config = get_config()
    sliding_window_size = config["model"]["mlp"]["sliding_window_size"]
    optimizer_name = config["training"]["mlp"]["optimizer"]
    loss_name = config["training"]["mlp"]["loss"]
    learning_rate = config["training"]["mlp"]["lr"]
    epochs = config["training"]["mlp"]["epoch"]
    precision = config["model"]["mlp"]["precision"]
    return_weights_for_loss = config["training"]["mlp"]["loss_weight"]

    # Fetching data
    inputs, labels, weights = fetch_data_for_training(stop=stop,return_weights_for_loss=return_weights_for_loss)

    train_dataloader, test_dataloader = preprocess_data_for_mlp(
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
    model = MlpSlidingWindow(window_size=sliding_window_size).to(device)
    if precision == "half":    
        model.to(torch.half)

    # TODO tune both of them to improve
    # Loss function 
    # weight = torch.tensor(
    #     [2.5, 3.8, 1.0, 3.8, 2.5, 3.2], dtype=torch.float32
    # ) 
    loss_params = {"weight":weights}
    print(f"Loss params : {loss_params}")
    loss_fn = get_loss(loss=loss_name,**loss_params)
    # Optimizer
    optim_params = {"lr":learning_rate}
    optimizer = get_optimizer(optim=optimizer_name,model_params=model.parameters(),**optim_params)

    # Training loop : TODO adding batch normalization + tuning epoch number
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device=device)
        test(test_dataloader, model, loss_fn, device=device)
    print("Done!")

    os.makedirs("weights/mlp_sw", exist_ok=True)
    torch.save(model.state_dict(), "weights/mlp_sw/mlp_sliding_window_model.pth")
    print("Saved PyTorch Model State to weights/mlp_sw/mlp_sliding_window_model.pth")


def train(dataloader, model, loss_fn, optimizer, device: str = "cpu"):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()  # remise a zero
        loss.backward()  # calcul des gradients
        optimizer.step()  # mise a jour des poids

        if batch % 100 == 0:
            loss_value, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device: str = "cpu"):
    num_batches = len(dataloader)
    model.eval()
    size = 0
    test_loss, correct = 0, 0
    metric = MulticlassF1Score(num_classes=6).to(device=device)
    f1score = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
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
