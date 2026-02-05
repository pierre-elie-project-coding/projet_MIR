from models.train_mlp import train_mlp

def train_models():
    print("="*150)
    print("Starting training \n")
    print("="*150)

    # Training the naive mlp
    precision = "half"
    print(f"Training Sliding window mlp in {precision} precision\n")
    train_mlp(batch_size=8,precision=precision,stop=64)
    print("="*150)
    print("Finished training")
    print("="*150)


if __name__=="__main__":
    train_models()
