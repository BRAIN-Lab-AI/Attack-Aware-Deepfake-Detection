import argparse
from unfooled.training.train import *
from unfooled.data.datasets import *
from unfooled.models.model import *

def main():
    p = argparse.ArgumentParser("UnFooled Training")
    p.add_argument("--data", type=str, required=False, default="data")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", type=str, default="auto")  # cpu|cuda|auto
    args = p.parse_args()
    print("Training entrypoint — wire your training loop here.")
    print(vars(args))

if __name__ == "__main__":
    main()
