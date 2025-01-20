from argparse import ArgumentParser
from os import path, makedirs
from torch import device, cuda

parser = ArgumentParser(
    description="Train a model to predict climate change opinions")
parser.add_argument("--crawl", type=str, choices=["twitter", "youtube", "bluesky"], default=None,
                    help="Platform to crawl")
parser.add_argument("--model", type=str, choices=["baseline", "blank", "sentiment"], default="baseline",
                    help="Model to train")
parser.add_argument("--sample", action="store_true",
                    help="Number of samples to use for training")
parser.add_argument("--epochs", type=int, default=10,
                    help="Number of epochs to train the model")
parser.add_argument("--batch", type=int, default=32,
                    help="Batch size for training")
parser.add_argument("--seed", type=int, default=42,
                    help="Seed for random number generation")
parser.add_argument("--data", type=str, default="res",
                    help="Path to the data directory")
parser.add_argument("--results", type=str, default=None,
                    help="Path to save the model, evaluation results and predictions")
parser.add_argument("--out", type=str, default=".out",
                    help="Path for generated files")
args = parser.parse_args()

BASE_PATH = path.dirname(__file__)
DATA_PATH = path.join(BASE_PATH, args.data)
OUT_PATH = path.join(BASE_PATH, args.out)
RESULTS_PATH = args.results or OUT_PATH
makedirs(RESULTS_PATH, exist_ok=True)
makedirs(OUT_PATH, exist_ok=True)

SEED = args.seed
EPOCHS = args.epochs
BATCH_SIZE = args.batch

SAMPLE = "sample-" if args.sample else ""
DEVICE = device("cuda" if cuda.is_available() else "cpu")

MODEL_NAME = args.model
if MODEL_NAME == "baseline":
    MODEL = "bert-base-uncased"
elif MODEL_NAME == "blank":
    MODEL = "bert-base-uncased"
elif MODEL_NAME == "sentiment":
    MODEL = "finiteautomata/bertweet-base-sentiment-analysis"

CRAWL_PLATFORM = args.crawl
