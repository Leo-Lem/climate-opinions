from argparse import ArgumentParser
from os import path

parser = ArgumentParser()
...
parser.add_argument("--data", type=str, default="res",
                    help="Path to the data directory")
args = parser.parse_args()

BASE_PATH = path.dirname(__file__)
DATA_PATH = path.join(BASE_PATH, args.data)
OUT_PATH = path.join(BASE_PATH, ".out")
