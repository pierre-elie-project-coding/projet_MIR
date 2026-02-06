from models.train_mlp import train_mlp
from utils.parse_config import get_config


def train_models():
    print("=" * 150)
    print("Starting training \n")
    print("=" * 150)

    # Training the naive mlp
    train_mlp(stop=1024)
    print("=" * 150)
    print("Finished training")
    print("=" * 150)


if __name__ == "__main__":
    train_models()
