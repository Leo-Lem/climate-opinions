# Details

## Folder structure
- [\_\_main\_\_.py](__main__.py) is the entry point for the program.
- [\_\_params\_\_.py](__params__.py) contains the hyperparameters, configuration and command line arguments.
- [src](src) is where the source code is stored.
  - [data.py](src/data.py) contains the data loading and preprocessing functions.
  - [train.py](src/train.py) contains the training loop.
  - [eval.py](src/eval.py) contains the evaluation functions.
  - [pred.py](src/pred.py) contains the prediction functions.
  - [visualize.py](src/visualize.py) contains the visualizations.
  - [crawl](src/crawl) contains the web crawling functions (bluesky, twitter, youtube).
- [res](res) is where the resources are stored.
- [.out](.out) is where the generated files are stored.
- [.devcontainer](.devcontainer) stores a VSCode devcontainer configuration.
- [requirements.txt](requirements.txt) lists the required packages with versions.

## Setup

```sh
pip install -r requirements.txt
```

## Data crawling
```sh
python climate-opinions --crawl=<platform> \
  --api_key=<api-key> \
  --query=<query> \
  --results=<results-directory>
```

Defaults are configured in [\_\_params\_\_.py](__params__.py).
- 'bluesky|twitter|youtube' (platform selection)
- api key for youtube.
- query (default is "global warming", "climate crisis", "climate emergency", "global heating", "climate change").
- results directory to store the crawled posts.

## Model training, prediction, and visualization
### Hyperparameters and configuration
```sh
python climate-opinions --model=<model> \
  --epochs <epochs> \
  --batch_size <batch_size> \
  --sample
  --results=<results-directory>
```

Defaults can be configured in [\_\_params\_\_.py](__params__.py).
- 'baseline|blank|sentiment' (model selection).
- epochs and batch size for training (defaults are 10 and 32).
- seed for random number generation (for reproducibility) (default: 42).
- results directory to store the best model, evaluation results and predictions.

#### Sample mode

When developing, you can use the `--sample` flag to only use a small subset of the data to see if everything works before starting long training processes.

### Data Preprocessing
- news label is removed.
- text is lowercased.
- links are removed.
- labels are shifted to 0, 1, 2 (easier to work with non-negative numbers)

### Model
- Blank and baseline are based on huggingface BERT-base-uncased.
- Sentiment model is based on BERTweet (BERTweet is a BERT model pre-trained on a large corpus of English tweets).

### Training
- epochs/batch size can be configured via command line arguments or in code.
- best model is selected by (validation) loss and saved.

### Evaluation
- best model is loaded.
- evaluation will be run if training is interrupted.
- evaluation metrics are: accuracy, precision/recall/f1 for each class (negative=0, neutral=1, positive=2).
- evaluation results are stored in csv with model name and metadata.

### Prediction
- best model is loaded.
- prediction is done on the crawled data.
- prediction results are stored in csv for selected model.

### Visualization
- visualize share of climate change opponents over time.
- …

# [README](README.md)