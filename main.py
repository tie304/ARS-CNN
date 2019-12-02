import argparse

from modules.train import train
from modules.evaluate import evaluate


parser = argparse.ArgumentParser()
parser.add_argument("--train", help="trains model on provided environment", action='store_true')
parser.add_argument("--evaluate", help="runs model on provided environment", type=int)
args = parser.parse_args()


if __name__ == "__main__":
    if args.train:
        train()
    if args.evaluate and args.evaluate > 1:
        evaluate(args.evaluate)