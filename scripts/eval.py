import argparse
from unfooled.training.train import *
from unfooled.data.datasets import *
from unfooled.models.model import *

def main():
    p = argparse.ArgumentParser("UnFooled Evaluation")
    p.add_argument("--data", type=str, required=False, default="data")
    p.add_argument("--checkpoint", type=str, required=False, default="ckpt.pt")
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()
    print("Evaluation entrypoint — wire your eval here.")
    print(vars(args))

if __name__ == "__main__":
    main()
