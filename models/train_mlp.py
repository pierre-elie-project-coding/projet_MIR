import torch
import torch.nn as nn
from models.mlp import MlpSlidingWindow
from data_process.mlp_preprocess_data import preprocess_data_for_mlp
# from data_process.process_data import fetch_data_for_training, tensor_to_dataloader
from data_process.process_data import fetch_data_for_training
from models.train import train
from models.test import test
import os

def train_mlp(batch_size:int=32,stop:int|None=None,precision:str="full"):

    # Fetching data
    inputs,labels = fetch_data_for_training(stop=stop)
    train_dataloader,test_dataloader = preprocess_data_for_mlp(input_tensor=inputs,
                                                               labels=labels,
                                                               batch_size=32,
                                                               with_split=0.8,
                                                               sliding_window_size=51,
                                                               shuffle=True)
    
    print(f"Train dataloader : {len(train_dataloader)}")
    print(f"Test dataloader : {len(test_dataloader)}")
    # Device
    device = (
        torch.accelerator.current_accelerator().type # type: ignore
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Instance of the mlp class
    model = MlpSlidingWindow().to(device)

    if precision=="half":
        model = model.half()
        
    # Loss function + optimizer : TODO tune both of them to improve
    weight = torch.tensor([6.7,14.5,1.0,14.5,6.4,10.0],dtype=torch.half) # TODO automate the calculation
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Training loop : TODO adding batch normalization + tuning epoch number
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer,device=device)
        test(test_dataloader, model, loss_fn,device=device)
    print("Done!")

    os.makedirs("weights/mlp_sw", exist_ok=True)
    torch.save(model.state_dict(), "weights/mlp_sw/mlp_sliding_window_model.pth")
    print("Saved PyTorch Model State to weights/mlp_sw/mlp_sliding_window_model.pth")
