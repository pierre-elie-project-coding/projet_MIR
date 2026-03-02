from models.train_mlp import train_mlp
from models.train_unet import train_unet
from models.train_xgboost import train_xgboost
from utils.parse_config import get_config
from utils.write_output import save_metrics

def train_models():
    config = get_config()
    model_to_train = config["train"]
    result_file = config["result_file"]
    stop = config["training"][model_to_train]["stop"]
    
    print("=" * 150)
    print("Starting training \n")
    print("=" * 150)

    if model_to_train == "mlp":
        # Training the naive mlp
        (epochs,loss,accuracy,f1score) = train_mlp(stop=stop)
        save_metrics(epochs=epochs,loss=loss,accuracy=accuracy,f1score=f1score)
    elif model_to_train == "unet":
        print(f"Stopping at {stop}") if stop else print(f"Using all dataset")
        (epochs,loss,accuracy,f1score) = train_unet(stop=stop)
        save_metrics(epochs=epochs,loss=loss,accuracy=accuracy,f1score=f1score)
    elif model_to_train == "xgboost":
        print(f"Stopping at {stop}") if stop else print(f"Using all dataset")
        (epochs,loss,accuracy,f1score) = train_xgboost(stop=stop)
        save_metrics(epochs=epochs,loss=loss,accuracy=accuracy,f1score=f1score)

    print("=" * 150)
    print("Finished training")
    print(f"See results at {result_file}") if result_file else None
    print("=" * 150)

if __name__ == "__main__":
    train_models()
